fn main() {
    let s = ctc_forced_aligner_wgpu::shaders::gemm_bias();
    let start = s.find("struct Dims").unwrap();
    let end = s[start..].find("}}").unwrap() + start + 2;
    println!("{}", &s[start..end]);
}
