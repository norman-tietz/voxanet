use crate::entity::Player;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GalaxyRequest {
    Home, // respawn on the start planet
    Goto {
        planet: usize,
        radii: f32,
    }, // 1-based planet number; distance from its centre in its radii
    Add {
        planet_type: crate::biome::PlanetType,
        radius: f32,
    }, // debug: append a planet to the galaxy
}

const GALAXY_USAGE: &str = "Usage: /galaxy home|goto <n> <radii>|add <type> <radius>";

// parses the words after "/galaxy"; the planet number's upper bound is checked by the game, which
// knows how many planets the galaxy has
fn parse_galaxy_command(args: &[&str]) -> Result<GalaxyRequest, &'static str> {
    match args {
        ["home"] => Ok(GalaxyRequest::Home),
        ["add", planet_type, radius] => {
            let planet_type = crate::biome::PlanetType::from_name(planet_type);
            let radius = radius.parse::<f32>().ok().filter(|r| {
                (crate::galaxy::MIN_ADDED_RADIUS..=crate::galaxy::MAX_ADDED_RADIUS).contains(r)
            });
            match (planet_type, radius) {
                (Some(planet_type), Some(radius)) => Ok(GalaxyRequest::Add {
                    planet_type,
                    radius,
                }),
                _ => Err("Usage: /galaxy add earthlike|volcanic|ice <radius 20-500>"),
            }
        }
        ["goto", n, radii] => {
            let planet = n.parse::<usize>().ok().filter(|&n| n >= 1);
            let radii = radii
                .parse::<f32>()
                .ok()
                .filter(|r| r.is_finite() && *r > 1.0);
            match (planet, radii) {
                (Some(planet), Some(radii)) => Ok(GalaxyRequest::Goto { planet, radii }),
                _ => Err("Usage: /galaxy goto <planet number> <distance in radii, more than 1>"),
            }
        }
        _ => Err(GALAXY_USAGE),
    }
}

pub struct Console {
    pub is_open: bool,
    pub input_buffer: String,
    pub history: Vec<(String, [f32; 3])>,
    pub height_fraction: f32,
    pub hw_shadows_request: Option<bool>, // set by /hw_shadows, applied by the game loop (renderer)
    pub screenshot_request: Option<String>, // set by /screenshot, applied by the game loop (renderer)
    pub view_request: Option<bool>, // set by /view, applied by the game loop (controller); true = first person
    pub galaxy_request: Option<GalaxyRequest>, // set by /galaxy, applied by the game loop
    command_history: Vec<String>, // submitted commands, oldest first (up/down), separate from the output log
    history_cursor: Option<usize>, // the entry up/down currently shows; None = not browsing

    history_capacity: usize,
}

impl Console {
    pub fn new() -> Self {
        Self {
            is_open: false,
            input_buffer: String::new(),
            history: Vec::new(),
            height_fraction: 0.0,
            hw_shadows_request: None,
            screenshot_request: None,
            view_request: None,
            galaxy_request: None,
            command_history: Vec::new(),
            history_cursor: None,
            history_capacity: 50,
        }
    }

    pub fn toggle(&mut self) {
        self.is_open = !self.is_open;
        if self.is_open {
            self.input_buffer.clear();
            self.history_cursor = None;
        }
    }

    pub fn log(&mut self, text: &str, color: [f32; 3]) {
        // print to actual terminal
        println!("{}", text);

        if self.history.len() >= self.history_capacity {
            self.history.remove(0);
        }
        self.history.push((text.to_string(), color));
    }

    // keeps a submitted command for up/down (not twice in a row; at most history_capacity) and ends
    // any browsing, so the next up starts again from the newest
    fn remember_command(&mut self, cmd: &str) {
        if self.command_history.last().map(String::as_str) != Some(cmd) {
            if self.command_history.len() >= self.history_capacity {
                self.command_history.remove(0);
            }
            self.command_history.push(cmd.to_string());
        }
        self.history_cursor = None;
    }

    // up: replace the input line with the next-older submitted command (stays at the oldest)
    pub fn history_up(&mut self) {
        if self.command_history.is_empty() {
            return;
        }
        let i = match self.history_cursor {
            None => self.command_history.len() - 1,
            Some(i) => i.saturating_sub(1),
        };
        self.history_cursor = Some(i);
        self.input_buffer = self.command_history[i].clone();
    }

    // down: the next-newer command; past the newest, an empty line again
    pub fn history_down(&mut self) {
        match self.history_cursor {
            None => {}
            Some(i) if i + 1 < self.command_history.len() => {
                self.history_cursor = Some(i + 1);
                self.input_buffer = self.command_history[i + 1].clone();
            }
            Some(_) => {
                self.history_cursor = None;
                self.input_buffer.clear();
            }
        }
    }

    pub fn handle_char(&mut self, c: char) {
        if !self.is_open {
            return;
        }
        // filter control characters
        if !c.is_control() {
            self.input_buffer.push(c);
        }
    }

    pub fn handle_backspace(&mut self) {
        if !self.is_open {
            return;
        }
        self.input_buffer.pop();
    }

    pub fn submit(&mut self, player: &mut Player) {
        if self.input_buffer.is_empty() {
            return;
        }

        let cmd = self.input_buffer.clone();
        self.remember_command(&cmd);
        self.log(&format!("> {}", cmd), [1.0, 1.0, 1.0]); // log

        self.process_command(&cmd, player);
        self.input_buffer.clear();
    }

    // scripted command injection (see main.rs's screenshot/command trigger files), bypassing the input buffer
    pub fn exec(&mut self, cmd_line: &str, player: &mut Player) {
        self.log(&format!("> {}", cmd_line), [1.0, 1.0, 1.0]);
        self.process_command(cmd_line, player);
    }

    fn process_command(&mut self, cmd_line: &str, player: &mut Player) {
        let parts: Vec<&str> = cmd_line.trim().split_whitespace().collect();
        if parts.is_empty() {
            return;
        }

        let command = parts[0];

        match command {
            "/move_speed" => {
                self.handle_property_command(parts, "move_speed", &mut player.move_speed);
            }
            "/jump_force" => {
                self.handle_property_command(parts, "jump_force", &mut player.jump_force);
            }

            "/debug_mode" => {
                if parts.len() < 3 || parts[1] != "set" {
                    self.log("Usage: /debug_mode set [true/false]", [1.0, 0.5, 0.0]);
                    return;
                }
                match parts[2] {
                    "true" => {
                        player.debug_mode = true;
                        self.log("Debug Mode: ON", [0.0, 1.0, 0.0]);
                    }
                    "false" => {
                        player.debug_mode = false;
                        self.log("Debug Mode: OFF", [1.0, 0.0, 0.0]);
                    }
                    _ => self.log("Value must be true or false", [1.0, 0.0, 0.0]),
                }
            }

            "/hw_shadows" => match (parts.get(1), parts.get(2)) {
                (Some(&"set"), Some(&"true")) => self.hw_shadows_request = Some(true),
                (Some(&"set"), Some(&"false")) => self.hw_shadows_request = Some(false),
                _ => self.log("Usage: /hw_shadows set [true/false]", [1.0, 0.5, 0.0]),
            },

            "/view" => match (parts.get(1), parts.get(2)) {
                (Some(&"set"), Some(&"first")) => self.view_request = Some(true),
                (Some(&"set"), Some(&"third")) => self.view_request = Some(false),
                _ => self.log("Usage: /view set first|third", [1.0, 0.5, 0.0]),
            },

            "/galaxy" => match parse_galaxy_command(&parts[1..]) {
                Ok(request) => self.galaxy_request = Some(request),
                Err(message) => self.log(message, [1.0, 0.5, 0.0]),
            },

            "/screenshot" => match parts.get(1) {
                Some(path) => {
                    self.screenshot_request = Some(path.to_string());
                    self.log("Capturing screenshot...", [0.0, 1.0, 0.0]);
                }
                None => self.log("Usage: /screenshot <path>", [1.0, 0.5, 0.0]),
            },

            "help" => {
                self.log("Available Commands:", [0.0, 1.0, 1.0]);
                self.log("  /debug_mode set true", [0.8, 0.8, 0.8]);
                self.log("  /move_speed set {value}", [0.8, 0.8, 0.8]);
                self.log("  /jump_force set {value}", [0.8, 0.8, 0.8]);
                self.log(
                    "  /hw_shadows set true|false  (hardware ray-traced shadows)",
                    [0.8, 0.8, 0.8],
                );
                self.log(
                    "  /screenshot <path>  (save the current frame as PNG)",
                    [0.8, 0.8, 0.8],
                );
                self.log("  /view set first|third", [0.8, 0.8, 0.8]);
                self.log(
                    "  /galaxy home|goto <n> <radii>|add <type> <radius>",
                    [0.8, 0.8, 0.8],
                );
            }
            _ => {
                self.log(&format!("Unknown command: {}", command), [1.0, 0.0, 0.0]);
            }
        }
    }

    fn handle_property_command(&mut self, parts: Vec<&str>, name: &str, property: &mut f32) {
        if parts.len() < 2 {
            self.log(&format!("Usage: /{} [set/get]", name), [1.0, 0.5, 0.0]);
            return;
        }

        match parts[1] {
            "get" => {
                self.log(
                    &format!("{} is currently: {:.2}", name, property),
                    [0.0, 1.0, 0.0],
                );
            }
            "set" => {
                if parts.len() < 3 {
                    self.log(&format!("Usage: /{} set <value>", name), [1.0, 0.5, 0.0]);
                    return;
                }
                match parts[2].parse::<f32>() {
                    Ok(val) => {
                        *property = val;
                        self.log(&format!("{} set to {:.2}", name, val), [0.0, 1.0, 0.0]);
                    }
                    Err(_) => {
                        self.log("Invalid number format.", [1.0, 0.0, 0.0]);
                    }
                }
            }
            _ => {
                self.log(
                    &format!("Unknown operation '{}'. Use set or get.", parts[1]),
                    [1.0, 0.5, 0.0],
                );
            }
        }
    }

    pub fn update_animation(&mut self, dt: f32) {
        let speed = 5.0;
        if self.is_open {
            self.height_fraction = (self.height_fraction + dt * speed).min(1.0);
        } else {
            self.height_fraction = (self.height_fraction - dt * speed).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn console_with(commands: &[&str]) -> Console {
        let mut c = Console::new();
        let mut p = Player::new();
        c.is_open = true;
        for cmd in commands {
            c.input_buffer = cmd.to_string();
            c.submit(&mut p);
        }
        c
    }

    #[test]
    fn up_and_down_step_through_submitted_commands() {
        let mut c = console_with(&["/a", "/b", "/c"]);
        c.history_up();
        assert_eq!(c.input_buffer, "/c");
        c.history_up();
        assert_eq!(c.input_buffer, "/b");
        c.history_up();
        c.history_up();
        assert_eq!(c.input_buffer, "/a", "stops at the oldest");
        c.history_down();
        assert_eq!(c.input_buffer, "/b");
        c.history_down();
        c.history_down();
        assert_eq!(c.input_buffer, "", "past the newest: an empty line");
    }

    #[test]
    fn repeated_commands_are_stored_once_and_submitting_resets_browsing() {
        let mut c = console_with(&["/a", "/a", "/b"]);
        c.history_up();
        c.history_up();
        assert_eq!(c.input_buffer, "/a");
        c.history_up();
        assert_eq!(c.input_buffer, "/a", "only two entries: /a, /b");
        c.input_buffer = "/x".to_string();
        c.submit(&mut Player::new());
        c.history_up();
        assert_eq!(c.input_buffer, "/x", "browsing restarts from the newest");
    }

    #[test]
    fn history_with_nothing_submitted_does_nothing() {
        let mut c = Console::new();
        c.is_open = true;
        c.history_up();
        c.history_down();
        assert_eq!(c.input_buffer, "");
        let mut c = console_with(&["/a"]);
        c.history_down(); // down before any up
        assert_eq!(c.input_buffer, "");
    }

    #[test]
    fn parses_goto() {
        assert_eq!(
            parse_galaxy_command(&["goto", "2", "3.5"]),
            Ok(GalaxyRequest::Goto {
                planet: 2,
                radii: 3.5
            })
        );
    }

    #[test]
    fn rejects_bad_goto_arguments() {
        assert!(parse_galaxy_command(&["goto"]).is_err());
        assert!(parse_galaxy_command(&["goto", "2"]).is_err());
        assert!(parse_galaxy_command(&["goto", "0", "3"]).is_err());
        assert!(parse_galaxy_command(&["goto", "x", "3"]).is_err());
        assert!(parse_galaxy_command(&["goto", "2", "1"]).is_err()); // inside / on the surface
        assert!(parse_galaxy_command(&["goto", "2", "0.5"]).is_err());
        assert!(parse_galaxy_command(&["goto", "2", "nan"]).is_err());
        assert!(parse_galaxy_command(&["goto", "2", "inf"]).is_err());
    }

    #[test]
    fn parses_home_goto_and_add() {
        assert_eq!(parse_galaxy_command(&["home"]), Ok(GalaxyRequest::Home));
        assert_eq!(
            parse_galaxy_command(&["goto", "2", "3.5"]),
            Ok(GalaxyRequest::Goto {
                planet: 2,
                radii: 3.5
            })
        );
        assert_eq!(
            parse_galaxy_command(&["add", "Ice", "120"]),
            Ok(GalaxyRequest::Add {
                planet_type: crate::biome::PlanetType::Ice,
                radius: 120.0
            })
        );
        // the home planet is gone: you leave and arrive by flying
        assert!(parse_galaxy_command(&["enter"]).is_err());
        assert!(parse_galaxy_command(&["exit"]).is_err());
        assert!(parse_galaxy_command(&["land", "3"]).is_err());
    }

    #[test]
    fn rejects_bad_add_arguments() {
        assert!(parse_galaxy_command(&["add"]).is_err());
        assert!(parse_galaxy_command(&["add", "ice"]).is_err());
        assert!(parse_galaxy_command(&["add", "lava", "100"]).is_err());
        assert!(parse_galaxy_command(&["add", "ice", "10"]).is_err());
        assert!(parse_galaxy_command(&["add", "ice", "600"]).is_err());
        assert!(parse_galaxy_command(&["add", "ice", "nan"]).is_err());
    }

    #[test]
    fn rejects_unknown_subcommands() {
        assert!(parse_galaxy_command(&["fly"]).is_err());
        assert!(parse_galaxy_command(&[]).is_err());
    }
}
