use crate::player::entity::Player;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SystemRequest {
    Home, // respawn on the start planet
    Goto {
        planet: usize,
        radii: f32,
    }, // 1-based planet number; distance from its centre in its radii
    Add {
        planet_type: crate::planet::biome::PlanetType,
        radius: f32,
    }, // debug: append a planet to the solar system
}

// every command with its options, shown when the console opens and by `help`; two per line so the
// whole list fits the quarter-screen console (about 8 lines at 800 px)
const HELP_LINES: [&str; 12] = [
    "Commands (up/down recall earlier ones):",
    "Commands without a value show the current setting:",
    "  /debug_mode [true|false]       /view [first|third]",
    "  /move_speed [<value>]          /jump_force [<value>]",
    "  /hw_shadows [true|false]       /style [hex|lowpoly|cubes]",
    "  /bevel [on|off|width <v>|cavity <v>]  /film [on|off|grain <v>|vignette <v>]",
    "  /ao [on|off|radius <v>|rays <n>|strength <v>|samples <n>|motion <fast> <slow>]",
    "  /plastic [on|off|roughness|specular|smudge|scratch|reflection <v>|env [on|off]]",
    "  /light [bounce <0-1>|wrap <0-0.5>]   /shadow [on|off|sun <0-10>]",
    "  /weather [on|off|auto|clear|rain|storm|snow|particles [on|off]]   /perf <seconds> [label]",
    "  /system home | goto <n> <radii> | add earthlike|volcanic|ice|desert|toxic|ocean|shattered <20-500>   /map [n]   /travel <n>",
    "  /screenshot <path>             help",
];
const HELP_HEADER_COLOR: [f32; 3] = [0.0, 1.0, 1.0];
const HELP_COLOR: [f32; 3] = [0.8, 0.8, 0.8];

const SYSTEM_USAGE: &str = "Usage: /system home|goto <n> <radii>|add <type> <radius>";

// parses the words after "/system"; the planet number's upper bound is checked by the game, which
// knows how many planets the solar system has
fn parse_system_command(args: &[&str]) -> Result<SystemRequest, &'static str> {
    match args {
        ["home"] => Ok(SystemRequest::Home),
        ["add", planet_type, radius] => {
            let planet_type = crate::planet::biome::PlanetType::from_name(planet_type);
            let radius = radius.parse::<f32>().ok().filter(|r| {
                (crate::space::solar_system::MIN_ADDED_RADIUS
                    ..=crate::space::solar_system::MAX_ADDED_RADIUS)
                    .contains(r)
            });
            match (planet_type, radius) {
                (Some(planet_type), Some(radius)) => Ok(SystemRequest::Add {
                    planet_type,
                    radius,
                }),
                _ => Err("Usage: /system add earthlike|volcanic|ice|desert|toxic|ocean|shattered <radius 20-500>"),
            }
        }
        ["goto", n, radii] => {
            let planet = n.parse::<usize>().ok().filter(|&n| n >= 1);
            let radii = radii
                .parse::<f32>()
                .ok()
                .filter(|r| r.is_finite() && *r > 1.0);
            match (planet, radii) {
                (Some(planet), Some(radii)) => Ok(SystemRequest::Goto { planet, radii }),
                _ => Err("Usage: /system goto <planet number> <distance in radii, more than 1>"),
            }
        }
        _ => Err(SYSTEM_USAGE),
    }
}

pub struct Console {
    pub is_open: bool,
    pub input_buffer: String,
    pub history: Vec<(String, [f32; 3])>,
    pub height_fraction: f32,
    pub hw_shadows_request: Option<Option<bool>>, // set by /hw_shadows (None inside: show), applied by the game loop (renderer)
    pub screenshot_request: Option<String>, // set by /screenshot, applied by the game loop (renderer)
    pub view_request: Option<Option<bool>>, // set by /view (None inside: show), applied by the game loop (controller); true = first person
    pub system_request: Option<SystemRequest>, // set by /system, applied by the game loop
    pub style_request: Option<crate::meshing::lowpoly::TerrainStyle>, // set by /style, applied by the game loop (renderer)
    pub bevel_request: Option<crate::meshing::bevel::BevelCommand>, // set by /bevel, applied by the game loop (renderer)
    pub film_request: Option<crate::render::film::FilmCommand>, // set by /film, applied by the game loop (renderer)
    pub ao_request: Option<crate::render::ao::AoCommand>, // set by /ao, applied by the game loop (renderer)
    pub plastic_request: Option<crate::render::plastic::PlasticCommand>, // set by /plastic, applied by the game loop (renderer)
    pub perf_request: Option<(f32, String)>, // set by /perf: seconds and label (perf.rs)
    pub travel_request: Option<usize>, // set by /travel <n>: fly to planet n's orbit (0-based)
    pub map_request: Option<Option<usize>>, // set by /map [n]: toggle the map, or open it with planet n selected
    pub weather_request: Option<crate::weather::WeatherCommand>, // set by /weather, applied by the game loop
    pub light_request: Option<crate::render::light::LightCommand>, // set by /light, applied by the game loop (renderer)
    pub shadow_request: Option<crate::render::shadow::ShadowCommand>, // set by /shadow, applied by the game loop (renderer)
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
            system_request: None,
            style_request: None,
            bevel_request: None,
            film_request: None,
            ao_request: None,
            plastic_request: None,
            perf_request: None,
            travel_request: None,
            map_request: None,
            weather_request: None,
            light_request: None,
            shadow_request: None,
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
            if !self.help_is_last() {
                self.show_help();
            }
        }
    }

    fn show_help(&mut self) {
        for (i, line) in HELP_LINES.iter().enumerate() {
            let color = if i == 0 {
                HELP_HEADER_COLOR
            } else {
                HELP_COLOR
            };
            self.log(line, color);
        }
    }

    // the log already ends with the command list (reopened without running anything): don't repeat it
    fn help_is_last(&self) -> bool {
        self.history.len() >= HELP_LINES.len()
            && self.history[self.history.len() - HELP_LINES.len()..]
                .iter()
                .zip(HELP_LINES)
                .all(|((text, _), line)| text == line)
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
                match parts.get(1..) {
                    Some([]) => {}
                    Some(["true"]) => player.debug_mode = true,
                    Some(["false"]) => player.debug_mode = false,
                    _ => {
                        self.log("Usage: /debug_mode [true|false]", [1.0, 0.5, 0.0]);
                        return;
                    }
                }
                if player.debug_mode {
                    self.log("Debug Mode: ON", [0.0, 1.0, 0.0]);
                } else {
                    self.log("Debug Mode: OFF", [1.0, 0.0, 0.0]);
                }
            }

            "/hw_shadows" => match parts.get(1..) {
                Some([]) => self.hw_shadows_request = Some(None),
                Some(["true"]) => self.hw_shadows_request = Some(Some(true)),
                Some(["false"]) => self.hw_shadows_request = Some(Some(false)),
                _ => self.log("Usage: /hw_shadows [true|false]", [1.0, 0.5, 0.0]),
            },

            "/bevel" => match crate::meshing::bevel::BevelCommand::parse(&parts[1..]) {
                Ok(cmd) => self.bevel_request = Some(cmd),
                Err(usage) => self.log(usage, [1.0, 0.5, 0.0]),
            },

            "/plastic" => match crate::render::plastic::PlasticCommand::parse(&parts[1..]) {
                Ok(cmd) => self.plastic_request = Some(cmd),
                Err(usage) => self.log(usage, [1.0, 0.5, 0.0]),
            },

            "/light" => match crate::render::light::LightCommand::parse(&parts[1..]) {
                Ok(cmd) => self.light_request = Some(cmd),
                Err(usage) => self.log(usage, [1.0, 0.5, 0.0]),
            },

            "/travel" => match parts.get(1).map(|n| n.parse::<usize>()) {
                Some(Ok(n)) if n >= 1 => self.travel_request = Some(n - 1),
                _ => self.log("Usage: /travel <planet number>", [1.0, 0.5, 0.0]),
            },

            "/map" => match parts.get(1).map(|n| n.parse::<usize>()) {
                None => self.map_request = Some(None),
                Some(Ok(n)) if n >= 1 => self.map_request = Some(Some(n - 1)),
                _ => self.log("Usage: /map [planet number]", [1.0, 0.5, 0.0]),
            },

            "/perf" => match crate::app::perf::parse_command(&parts[1..]) {
                Ok(req) => self.perf_request = Some(req),
                Err(usage) => self.log(usage, [1.0, 0.5, 0.0]),
            },

            "/weather" => match crate::weather::WeatherCommand::parse(&parts[1..]) {
                Ok(cmd) => self.weather_request = Some(cmd),
                Err(usage) => self.log(usage, [1.0, 0.5, 0.0]),
            },

            "/shadow" => match crate::render::shadow::ShadowCommand::parse(&parts[1..]) {
                Ok(cmd) => self.shadow_request = Some(cmd),
                Err(usage) => self.log(usage, [1.0, 0.5, 0.0]),
            },

            "/ao" => match crate::render::ao::AoCommand::parse(&parts[1..]) {
                Ok(cmd) => self.ao_request = Some(cmd),
                Err(usage) => self.log(usage, [1.0, 0.5, 0.0]),
            },

            "/film" => match crate::render::film::FilmCommand::parse(&parts[1..]) {
                Ok(cmd) => self.film_request = Some(cmd),
                Err(usage) => self.log(usage, [1.0, 0.5, 0.0]),
            },

            "/view" => match parts.get(1..) {
                Some([]) => self.view_request = Some(None),
                Some(["first"]) => self.view_request = Some(Some(true)),
                Some(["third"]) => self.view_request = Some(Some(false)),
                _ => self.log("Usage: /view [first|third]", [1.0, 0.5, 0.0]),
            },

            "/style" => match parts.get(1) {
                None => self.log(
                    &format!("Terrain style: {:?}", crate::meshing::lowpoly::style()),
                    [0.0, 1.0, 0.0],
                ),
                Some(&"cubes") => {
                    self.style_request = Some(crate::meshing::lowpoly::TerrainStyle::Cubes)
                }
                Some(&"lowpoly") => {
                    self.style_request = Some(crate::meshing::lowpoly::TerrainStyle::LowPoly)
                }
                Some(&"hex") => {
                    self.style_request = Some(crate::meshing::lowpoly::TerrainStyle::Hex)
                }
                _ => self.log("Usage: /style [hex|lowpoly|cubes]", [1.0, 0.5, 0.0]),
            },

            "/system" => match parse_system_command(&parts[1..]) {
                Ok(request) => self.system_request = Some(request),
                Err(message) => self.log(message, [1.0, 0.5, 0.0]),
            },

            "/screenshot" => match parts.get(1) {
                Some(path) => {
                    self.screenshot_request = Some(path.to_string());
                    self.log("Capturing screenshot...", [0.0, 1.0, 0.0]);
                }
                None => self.log("Usage: /screenshot <path>", [1.0, 0.5, 0.0]),
            },

            "help" => self.show_help(),
            _ => {
                self.log(&format!("Unknown command: {}", command), [1.0, 0.0, 0.0]);
            }
        }
    }

    // /move_speed and the like: no value shows it, a value sets it
    fn handle_property_command(&mut self, parts: Vec<&str>, name: &str, property: &mut f32) {
        match parts.get(1..) {
            Some([]) => self.log(&format!("{name}: {property:.2}"), [0.0, 1.0, 0.0]),
            Some([v]) => match v.parse::<f32>() {
                Ok(val) => {
                    *property = val;
                    self.log(&format!("{name} set to {val:.2}"), [0.0, 1.0, 0.0]);
                }
                Err(_) => self.log("Invalid number format.", [1.0, 0.0, 0.0]),
            },
            _ => self.log(&format!("Usage: /{name} [<value>]"), [1.0, 0.5, 0.0]),
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

    // no value shows the current setting, a value sets it
    #[test]
    fn commands_without_a_value_show_the_setting() {
        let mut player = Player::new();
        let mut c = Console::new();
        c.exec("/move_speed", &mut player);
        assert!(c.history.last().unwrap().0.starts_with("move_speed: "));
        c.exec("/move_speed 7", &mut player);
        assert_eq!(player.move_speed, 7.0);
        c.exec("/move_speed set 7", &mut player);
        assert_eq!(c.history.last().unwrap().0, "Usage: /move_speed [<value>]");
        c.exec("/debug_mode true", &mut player);
        assert!(player.debug_mode);
        c.exec("/debug_mode", &mut player);
        assert_eq!(c.history.last().unwrap().0, "Debug Mode: ON");
        c.exec("/hw_shadows", &mut player);
        assert_eq!(c.hw_shadows_request.take(), Some(None));
        c.exec("/hw_shadows false", &mut player);
        assert_eq!(c.hw_shadows_request.take(), Some(Some(false)));
        c.exec("/view", &mut player);
        assert_eq!(c.view_request.take(), Some(None));
        c.exec("/plastic", &mut player);
        assert_eq!(
            c.plastic_request,
            Some(crate::render::plastic::PlasticCommand::Get)
        );
        c.exec("/light", &mut player);
        assert_eq!(
            c.light_request,
            Some(crate::render::light::LightCommand::Get)
        );
    }

    #[test]
    fn shadow_command_sets_a_request() {
        let c = console_with(&["/shadow sun 4"]);
        assert_eq!(
            c.shadow_request,
            Some(crate::render::shadow::ShadowCommand::Sun(4.0))
        );
        let c = console_with(&["/shadow sun 40"]);
        assert_eq!(c.shadow_request, None);
        assert_eq!(
            c.history.last().unwrap().0,
            crate::render::shadow::SHADOW_USAGE
        );
    }

    #[test]
    fn weather_command_sets_a_request() {
        let c = console_with(&["/weather storm"]);
        assert_eq!(
            c.weather_request,
            Some(crate::weather::WeatherCommand::Force(
                crate::weather::Force::Storm
            ))
        );
        let c = console_with(&["/weather"]);
        assert_eq!(
            c.weather_request,
            Some(crate::weather::WeatherCommand::Show)
        );
        let c = console_with(&["/weather hail"]);
        assert_eq!(c.weather_request, None);
    }

    #[test]
    fn light_command_sets_a_request() {
        let c = console_with(&["/light wrap 0.2"]);
        assert_eq!(
            c.light_request,
            Some(crate::render::light::LightCommand::Wrap(0.2))
        );
        let c = console_with(&["/light wrap 2"]);
        assert_eq!(c.light_request, None);
        assert_eq!(
            c.history.last().unwrap().0,
            crate::render::light::LIGHT_USAGE
        );
    }

    #[test]
    fn plastic_command_sets_a_request() {
        let c = console_with(&["/plastic roughness 0.5"]);
        assert_eq!(
            c.plastic_request,
            Some(crate::render::plastic::PlasticCommand::Roughness(0.5))
        );
        let c = console_with(&["/plastic specular 9"]);
        assert_eq!(c.plastic_request, None);
        assert_eq!(
            c.history.last().unwrap().0,
            crate::render::plastic::PLASTIC_USAGE
        );
    }

    #[test]
    fn ao_command_sets_a_request() {
        let c = console_with(&["/ao rays 8"]);
        assert_eq!(c.ao_request, Some(crate::render::ao::AoCommand::Rays(8)));
        let c = console_with(&["/ao rays 99"]);
        assert_eq!(c.ao_request, None);
        assert_eq!(c.history.last().unwrap().0, crate::render::ao::AO_USAGE);
    }

    #[test]
    fn film_command_sets_a_request() {
        let c = console_with(&["/film vignette 0.4"]);
        assert_eq!(
            c.film_request,
            Some(crate::render::film::FilmCommand::Vignette(0.4))
        );
        let c = console_with(&["/film grain 1"]);
        assert_eq!(c.film_request, None);
        assert_eq!(c.history.last().unwrap().0, crate::render::film::FILM_USAGE);
    }

    #[test]
    fn bevel_command_sets_a_request() {
        let c = console_with(&["/bevel width 0.08"]);
        assert_eq!(
            c.bevel_request,
            Some(crate::meshing::bevel::BevelCommand::Width(0.08))
        );
        let c = console_with(&["/bevel width 9"]);
        assert_eq!(c.bevel_request, None);
        assert_eq!(
            c.history.last().unwrap().0,
            crate::meshing::bevel::BEVEL_USAGE
        );
    }

    #[test]
    fn opening_the_console_lists_the_commands_once() {
        let mut c = Console::new();
        c.toggle();
        assert_eq!(c.history.len(), HELP_LINES.len());
        assert_eq!(c.history[0].0, HELP_LINES[0]);
        c.toggle();
        c.toggle(); // reopened with nothing run in between: not listed again
        assert_eq!(c.history.len(), HELP_LINES.len());
        c.exec("/view third", &mut Player::new());
        c.toggle();
        c.toggle(); // something else was logged since: listed again, at the bottom
        assert!(c.help_is_last());
        assert_eq!(c.history.len(), 2 * HELP_LINES.len() + 1);
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
            parse_system_command(&["goto", "2", "3.5"]),
            Ok(SystemRequest::Goto {
                planet: 2,
                radii: 3.5
            })
        );
    }

    #[test]
    fn rejects_bad_goto_arguments() {
        assert!(parse_system_command(&["goto"]).is_err());
        assert!(parse_system_command(&["goto", "2"]).is_err());
        assert!(parse_system_command(&["goto", "0", "3"]).is_err());
        assert!(parse_system_command(&["goto", "x", "3"]).is_err());
        assert!(parse_system_command(&["goto", "2", "1"]).is_err()); // inside / on the surface
        assert!(parse_system_command(&["goto", "2", "0.5"]).is_err());
        assert!(parse_system_command(&["goto", "2", "nan"]).is_err());
        assert!(parse_system_command(&["goto", "2", "inf"]).is_err());
    }

    #[test]
    fn parses_home_goto_and_add() {
        assert_eq!(parse_system_command(&["home"]), Ok(SystemRequest::Home));
        assert_eq!(
            parse_system_command(&["goto", "2", "3.5"]),
            Ok(SystemRequest::Goto {
                planet: 2,
                radii: 3.5
            })
        );
        assert_eq!(
            parse_system_command(&["add", "Ice", "120"]),
            Ok(SystemRequest::Add {
                planet_type: crate::planet::biome::PlanetType::Ice,
                radius: 120.0
            })
        );
        // the home planet is gone: you leave and arrive by flying
        assert!(parse_system_command(&["enter"]).is_err());
        assert!(parse_system_command(&["exit"]).is_err());
        assert!(parse_system_command(&["land", "3"]).is_err());
    }

    #[test]
    fn rejects_bad_add_arguments() {
        assert!(parse_system_command(&["add"]).is_err());
        assert!(parse_system_command(&["add", "ice"]).is_err());
        assert!(parse_system_command(&["add", "lava", "100"]).is_err());
        assert!(parse_system_command(&["add", "ice", "10"]).is_err());
        assert!(parse_system_command(&["add", "ice", "600"]).is_err());
        assert!(parse_system_command(&["add", "ice", "nan"]).is_err());
    }

    #[test]
    fn rejects_unknown_subcommands() {
        assert!(parse_system_command(&["fly"]).is_err());
        assert!(parse_system_command(&[]).is_err());
    }

    // /style switches between the terrain styles
    #[test]
    fn style_command_requests_a_style() {
        use crate::meshing::lowpoly::TerrainStyle;
        let mut c = Console::new();
        let mut player = crate::player::entity::Player::new();
        c.exec("/style cubes", &mut player);
        assert_eq!(c.style_request.take(), Some(TerrainStyle::Cubes));
        c.exec("/style lowpoly", &mut player);
        assert_eq!(c.style_request.take(), Some(TerrainStyle::LowPoly));
        c.exec("/style hex", &mut player);
        assert_eq!(c.style_request.take(), Some(TerrainStyle::Hex));
        c.exec("/style round", &mut player);
        assert_eq!(c.style_request, None);
    }
}
