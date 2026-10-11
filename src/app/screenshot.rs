// screenshot.rs
// Screen capture, triggered by F2 (src/main.rs) or the console's `/screenshot <path>` (src/cmd.rs).
// Writes a dependency-free PNG (uncompressed "stored" deflate blocks, so no zlib/png crate is needed) of
// the swapchain texture, downsampled so the file stays small. Blocks the frame it runs on; that's fine
// for an occasional user-triggered capture but not something to call every frame.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB88320 & mask);
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

fn push_chunk(out: &mut Vec<u8>, tag: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut body = Vec::with_capacity(4 + data.len());
    body.extend_from_slice(tag);
    body.extend_from_slice(data);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_be_bytes());
}

// zlib-wraps `raw` as uncompressed ("stored") deflate blocks, up to 65535 bytes each
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 8);
    out.push(0x78);
    out.push(0x01);
    let mut i = 0;
    loop {
        let end = (i + 65535).min(raw.len());
        let chunk = &raw[i..end];
        let is_last = end == raw.len();
        out.push(if is_last { 1 } else { 0 });
        out.extend_from_slice(&(chunk.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(chunk.len() as u16)).to_le_bytes());
        out.extend_from_slice(chunk);
        i = end;
        if is_last {
            break;
        }
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

fn encode_rgba8(width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n']);

    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, colour type 6 (RGBA), default compression/filter/interlace
    push_chunk(&mut out, b"IHDR", &ihdr);

    let stride = (width * 4) as usize;
    let mut raw = Vec::with_capacity((stride + 1) * height as usize);
    for row in 0..height as usize {
        raw.push(0); // filter type: None
        raw.extend_from_slice(&pixels[row * stride..row * stride + stride]);
    }
    push_chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    push_chunk(&mut out, b"IEND", &[]);
    out
}

// nearest-neighbour downsample to at most `max_dim` on the long side, unpadding rows and swapping
// BGRA -> RGBA if the swapchain format needs it; alpha is forced opaque (the swapchain's is meaningless)
fn downsample_rgba(
    src: &[u8],
    width: u32,
    height: u32,
    padded_row: u32,
    bgra: bool,
    max_dim: u32,
) -> (u32, u32, Vec<u8>) {
    let scale = (max_dim as f32 / width.max(height) as f32).min(1.0);
    let out_w = ((width as f32 * scale).round() as u32).max(1);
    let out_h = ((height as f32 * scale).round() as u32).max(1);
    let mut out = vec![0u8; (out_w * out_h * 4) as usize];
    for oy in 0..out_h {
        let sy = (((oy as f32 + 0.5) / out_h as f32) * height as f32) as u32;
        let sy = sy.min(height - 1);
        for ox in 0..out_w {
            let sx = (((ox as f32 + 0.5) / out_w as f32) * width as f32) as u32;
            let sx = sx.min(width - 1);
            let si = (sy * padded_row + sx * 4) as usize;
            let di = ((oy * out_w + ox) * 4) as usize;
            if bgra {
                out[di] = src[si + 2];
                out[di + 1] = src[si + 1];
                out[di + 2] = src[si];
            } else {
                out[di..di + 3].copy_from_slice(&src[si..si + 3]);
            }
            out[di + 3] = 255;
        }
    }
    (out_w, out_h, out)
}

// copies `texture` (the current swapchain texture, still valid until it's presented) to a PNG at `path`
pub fn capture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    path: &str,
) {
    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = unpadded.div_ceil(align) * align;

    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Screenshot Readback"),
        size: padded as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Screenshot Copy"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));

    let mapped = Arc::new(AtomicBool::new(false));
    let mapped2 = mapped.clone();
    buf.map_async(wgpu::MapMode::Read, .., move |result| {
        if result.is_ok() {
            mapped2.store(true, Ordering::Release);
        }
    });
    loop {
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if mapped.load(Ordering::Acquire) {
            break;
        }
    }

    let bgra = matches!(
        format,
        wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    let (out_w, out_h, rgba) = match buf.get_mapped_range(..) {
        Ok(view) => downsample_rgba(&view, width, height, padded, bgra, 1280),
        Err(e) => {
            println!("screenshot: readback map failed: {e}");
            return;
        }
    };
    buf.unmap();

    match std::fs::write(path, encode_rgba8(out_w, out_h, &rgba)) {
        Ok(()) => println!("screenshot saved to {path} ({out_w}x{out_h})"),
        Err(e) => println!("screenshot: failed to write {path}: {e}"),
    }
}
