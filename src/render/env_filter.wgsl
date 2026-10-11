// env_filter.wgsl
// The environment map's levels (env_map.rs): fs_env_mip box-filters the source cube's mip chain;
// fs_env_filter writes one level of a slot, GGX-filtered for roughness fp.roughness from the source
// (importance-sampled in direction space, V = N = R; the PDF picks a coarser source mip against
// undersampling — the standard pre-filtered environment), level 0 a copy. The face comes in as the
// instance index.

struct FilterParams {
    roughness: f32, // GGX roughness of the level being written (env_map.rs level_roughness)
    size: f32,      // its texels per face edge
    src_size: f32,  // the source's level-0 size (env_map.rs ENV_SIZE)
    _pad: f32,
}
@group(0) @binding(0) var src_cube: texture_cube<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var<uniform> fp: FilterParams;
@group(0) @binding(3) var mip_src: texture_2d<f32>; // fs_env_mip: the previous mip of the same face

const SAMPLES: u32 = 32u; // env_map.rs FILTER_SAMPLES
const PI: f32 = 3.14159265;

struct FaceOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) face: u32,
}

@vertex
fn vs_env_face(@builtin(vertex_index) i: u32, @builtin(instance_index) face: u32) -> FaceOut {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var out: FaceOut;
    out.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.face = face;
    return out;
}

// shader.wgsl env_cube_dir
fn cube_dir(face: u32, px: vec2<f32>, size: f32) -> vec3<f32> {
    let st = px / size * 2.0 - 1.0;
    let s = st.x;
    let t = st.y;
    switch face {
        case 0u: { return vec3<f32>(1.0, -t, -s); }
        case 1u: { return vec3<f32>(-1.0, -t, s); }
        case 2u: { return vec3<f32>(s, 1.0, t); }
        case 3u: { return vec3<f32>(s, -1.0, -t); }
        case 4u: { return vec3<f32>(s, -t, 1.0); }
        default: { return vec3<f32>(-s, -t, -1.0); }
    }
}

fn hammersley(i: u32, n: u32) -> vec2<f32> {
    var b = i;
    b = (b << 16u) | (b >> 16u);
    b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
    b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
    b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
    b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
    return vec2<f32>(f32(i) / f32(n), f32(b) * 2.3283064365386963e-10);
}

@fragment
fn fs_env_mip(in: FaceOut) -> @location(0) vec4<f32> {
    let p = vec2<i32>(in.pos.xy) * 2;
    return 0.25 * (textureLoad(mip_src, p, 0) + textureLoad(mip_src, p + vec2<i32>(1, 0), 0)
        + textureLoad(mip_src, p + vec2<i32>(0, 1), 0) + textureLoad(mip_src, p + vec2<i32>(1, 1), 0));
}

@fragment
fn fs_env_filter(in: FaceOut) -> @location(0) vec4<f32> {
    let N = normalize(cube_dir(in.face, in.pos.xy, fp.size));
    if (fp.roughness <= 0.0) {
        return textureSampleLevel(src_cube, src_sampler, N, 0.0);
    }
    let al = fp.roughness * fp.roughness;
    let a2 = al * al;
    // a tangent frame around N
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(N.z) < 0.999);
    let tx = normalize(cross(helper, N));
    let ty = cross(N, tx);
    let sa_texel = 4.0 * PI / (6.0 * fp.src_size * fp.src_size);
    var sum = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var i = 0u; i < SAMPLES; i++) {
        let xi = hammersley(i, SAMPLES);
        // GGX half vector around N
        let phi = 2.0 * PI * xi.x;
        let cos_t = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
        let sin_t = sqrt(1.0 - cos_t * cos_t);
        let H = normalize(tx * (cos(phi) * sin_t) + ty * (sin(phi) * sin_t) + N * cos_t);
        let Ld = normalize(2.0 * dot(N, H) * H - N);
        let n_l = dot(N, Ld);
        if (n_l > 0.0) {
            // PDF (V = N: n·h = h·v): D / 4; one sample covers 1 / (SAMPLES · pdf) steradians
            let dd = cos_t * cos_t * (a2 - 1.0) + 1.0;
            let D = a2 / (PI * dd * dd);
            let sa_sample = 1.0 / (f32(SAMPLES) * D * 0.25 + 1e-4);
            let mip = max(0.5 * log2(sa_sample / sa_texel), 0.0);
            sum += textureSampleLevel(src_cube, src_sampler, Ld, mip) * n_l;
            wsum += n_l;
        }
    }
    return sum / max(wsum, 1e-4);
}
