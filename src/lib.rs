use lazy_static::lazy_static;

lazy_static! {
    static ref LOG: u64 = init_log();
}

// install a logger facility
fn init_log() -> u64 {
    let _res = env_logger::try_init();
    println!("\n ************** initializing logger *****************\n");
    1
}

pub mod merit;
pub mod rhst;
pub mod smalld;
