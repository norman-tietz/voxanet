// gpu_timer.rs
// GPU time per render pass from timestamp queries, averaged over one second for the debug overlay.
// Readback is asynchronous: frames rendered while a readback is in flight are simply not sampled.
//
// Apple GPUs overlap passes: a pass's begin timestamp is taken when its vertex work starts, often while
// the previous pass is still shading, so begin-to-end intervals overlap and include waiting. The parts
// form a dependency chain (G-buffer -> rays -> blur -> main reads the blur -> text draws over main), so their
// fragment work is serialised and a part's time is measured from the previous part's end to its own end.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

// measured parts of a frame; each has a begin and an end timestamp (query 2i and 2i + 1)
pub const PARTS: [&str; 5] = ["G-buffer", "Rays", "Blur", "Main", "Text"];
pub const GBUFFER: usize = 0; // shadow G-buffer pass
pub const RAYS: usize = 1;    // shadow compute pass (ray march or hardware rays)
pub const BLUR: usize = 2;
pub const MAIN: usize = 3;
pub const TEXT: usize = 4;

const QUERIES: u32 = 2 * PARTS.len() as u32;
const BYTES: u64 = QUERIES as u64 * wgpu::QUERY_SIZE as u64;

pub struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    mapped: Arc<AtomicBool>, // set by the map_async callback
    in_flight: bool,         // readback_buf is being mapped, don't copy into it
    copied: bool,            // this frame's encoder copies into readback_buf
    period_ns: f32,
    sums_ms: [f64; PARTS.len()],
    samples: u32,
    window_start: Instant,
    pub average_ms: [f32; PARTS.len()],
}

impl GpuTimer {
    pub fn required_features(adapter: &wgpu::Adapter) -> wgpu::Features {
        adapter.features() & wgpu::Features::TIMESTAMP_QUERY
    }

    // None when the device has no timestamp queries
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("GPU Timer"), ty: wgpu::QueryType::Timestamp, count: QUERIES });
        let buffer = |label, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: BYTES, usage, mapped_at_creation: false });
        Some(Self {
            query_set,
            resolve_buf: buffer("GPU Timer Resolve", wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            readback_buf: buffer("GPU Timer Readback", wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
            mapped: Arc::new(AtomicBool::new(false)),
            in_flight: false,
            copied: false,
            period_ns: queue.get_timestamp_period(),
            sums_ms: [0.0; PARTS.len()],
            samples: 0,
            window_start: Instant::now(),
            average_ms: [0.0; PARTS.len()],
        })
    }

    // timestamp writes for a render pass: the begin and/or end of `part`
    pub fn writes(&self, part: usize, begin: bool, end: bool) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: begin.then_some(2 * part as u32),
            end_of_pass_write_index: end.then_some(2 * part as u32 + 1),
        })
    }

    pub fn compute_writes(&self, part: usize, begin: bool, end: bool) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: begin.then_some(2 * part as u32),
            end_of_pass_write_index: end.then_some(2 * part as u32 + 1),
        })
    }

    // call after all passes of the frame are encoded
    pub fn resolve(&mut self, enc: &mut wgpu::CommandEncoder) {
        enc.resolve_query_set(&self.query_set, 0..QUERIES, &self.resolve_buf, 0);
        self.copied = !self.in_flight;
        if self.copied {
            enc.copy_buffer_to_buffer(&self.resolve_buf, 0, &self.readback_buf, 0, BYTES);
        }
    }

    // call after the frame is submitted
    pub fn after_submit(&mut self) {
        if !self.copied { return; }
        self.in_flight = true;
        let mapped = self.mapped.clone();
        self.readback_buf.map_async(wgpu::MapMode::Read, .., move |result| {
            if result.is_ok() { mapped.store(true, Ordering::Release); }
        });
    }

    // call once per frame: collects a finished readback and updates the averages every second
    pub fn poll(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        if self.in_flight && self.mapped.swap(false, Ordering::Acquire) {
            if let Ok(view) = self.readback_buf.get_mapped_range(..) {
                let ticks: &[u64] = bytemuck::cast_slice(&view);
                for part in 0..PARTS.len() {
                    let start = if part == 0 { ticks[0] } else { ticks[2 * part - 1] }; // previous part's end
                    self.sums_ms[part] += ticks[2 * part + 1].saturating_sub(start) as f64 * self.period_ns as f64 / 1e6;
                }
                self.samples += 1;
            }
            self.readback_buf.unmap();
            self.in_flight = false;
        }
        if self.window_start.elapsed().as_secs_f32() >= 1.0 && self.samples > 0 {
            for part in 0..PARTS.len() {
                self.average_ms[part] = (self.sums_ms[part] / self.samples as f64) as f32;
            }
            self.sums_ms = [0.0; PARTS.len()];
            self.samples = 0;
            self.window_start = Instant::now();
        }
    }

    pub fn summary(&self) -> String {
        let total: f32 = self.average_ms.iter().sum();
        let parts: Vec<String> = PARTS.iter().zip(self.average_ms).map(|(n, ms)| format!("{n:<8}{ms:6.2} ms")).collect();
        format!("GPU     {total:6.2} ms\n{}", parts.join("\n"))
    }
}
