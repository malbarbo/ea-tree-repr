use pbr::ProgressBar;

use std::io::{stderr, Stderr};
use std::time::Duration;

pub struct ProgressIter<I> {
    iter: I,
    pb: ProgressBar<Stderr>,
}

pub fn progress<I: ExactSizeIterator>(iter: I) -> ProgressIter<I> {
    let len = iter.len() as u64;
    let mut pb = ProgressBar::on(stderr(), len);
    pb.show_time_left = true;
    pb.set_max_refresh_rate(Some(Duration::from_millis(100)));
    ProgressIter { iter, pb }
}

impl<I: Iterator> Iterator for ProgressIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.iter.next() {
            self.pb.inc();
            Some(item)
        } else {
            self.pb.set_max_refresh_rate(None);
            self.pb.finish();
            None
        }
    }
}
