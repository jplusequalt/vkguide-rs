#![allow(dead_code)]

mod engine;

use log::*;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowAttributes, WindowId},
};

use crate::engine::engine::Engine;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;

pub struct App {
    pub window: Option<Window>,
    engine: Option<Engine>,
    minimized: bool,
    resized: bool,
}

impl App {
    pub fn new() -> Self {
        Self {
            window: None,
            engine: None,
            minimized: false,
            resized: false,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                WindowAttributes::default()
                    .with_title("vk-guide")
                    .with_inner_size(LogicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        let engine = Engine::create(&window).unwrap();
        
        self.window = Some(window);
        self.engine = Some(engine);

    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {      
        match event {
            WindowEvent::CloseRequested => {
                info!("Exiting ...");
                self.engine.as_mut().map(|e| e.destroy());
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.engine
                    .as_mut()
                    .map(|e| e.render(self.window.as_ref().unwrap()));
            }
            _ => (),
        }
    }
}
