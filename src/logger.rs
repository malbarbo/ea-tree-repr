// system
use std::str::FromStr;

// external
use log::{self, LevelFilter, Log, Metadata, Record};

pub struct Logger {
    level: LevelFilter,
}

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.level() <= self.level
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            eprintln!("{}: {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

pub fn init_logger(level: &str) {
    if let Ok(level) = LevelFilter::from_str(level) {
        log::set_max_level(level);
        log::set_boxed_logger(Box::new(Logger { level })).unwrap();
        debug!("Logging at {:?} level", level);
    }
}
