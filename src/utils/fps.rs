#[derive(Default)]
pub struct FPSCounter {
    frame_times: Vec<f32>,
    max_samples: usize,
}

impl FPSCounter {
    pub fn new(max_samples: usize) -> Self {
        Self {
            frame_times: Vec::new(),
            max_samples,
        }
    }

    pub fn add_frame_time(&mut self, frame_time_ms: f32) {
        self.frame_times.push(frame_time_ms);
        if self.frame_times.len() > self.max_samples {
            self.frame_times.remove(0);
        }
    }

    pub fn get_fps(&self) -> f32 {
        if self.frame_times.len() == 0 {
            return 0.0;
        }

        let average_frame_time: f32 =
            self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
        1000.0 / average_frame_time
    }
}
