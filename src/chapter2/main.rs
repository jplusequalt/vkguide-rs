use anyhow::Result;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::engine::App;

mod engine;

fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();

    event_loop.run_app(&mut app)?;

    Ok(())
}
