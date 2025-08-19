"""
酒吧场景渲染器 - 文本界面显示酒吧场景
"""
import json
import os
from typing import Dict, List, Tuple
from colorama import init, Fore, Back, Style
from bar_agents import BarAgent, BarSimulation

init(autoreset=True)  # 初始化colorama

class BarRenderer:
    """酒吧场景渲染器"""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.room = self.config["room"]
        self.tiles = self.room["tiles"]
        self.legend = self.room["legend"]
        self.spawn_points = self.room["spawn_points"]
        self.agent_positions = {}
        
        # Character mapping table
        self.char_map = {
            "wall_brick": "#",
            "door": "+",
            "bar_counter": "=",
            "bar_stool": "~",
            "table": "O",
            "chair": "o",
            "music_stage": "M",
            "floor_wood": "."
        }
        
        # 颜色映射
        self.color_map = {
            "wall_brick": Fore.RED + Style.BRIGHT,
            "door": Fore.YELLOW + Style.BRIGHT,
            "bar_counter": Fore.CYAN + Style.BRIGHT,
            "bar_stool": Fore.MAGENTA,
            "table": Fore.GREEN,
            "chair": Fore.GREEN + Style.DIM,
            "music_stage": Fore.BLUE + Style.BRIGHT,
            "floor_wood": Fore.WHITE + Style.DIM
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def update_agent_positions(self, agents: Dict[str, BarAgent]):
        """更新代理位置"""
        self.agent_positions = {name: agent.position for name, agent in agents.items()}
    
    def get_char_at_position(self, x: int, y: int) -> Tuple[str, str]:
        """获取指定位置的字符和颜色"""
        # 检查是否有代理在此位置
        for name, pos in self.agent_positions.items():
            if pos == [x, y]:
                return name[0].upper(), Fore.WHITE + Back.RED + Style.BRIGHT
        
        # 获取地图字符
        if 0 <= y < len(self.tiles) and 0 <= x < len(self.tiles[y]):
            tile_char = self.tiles[y][x]
            tile_type = self.legend.get(tile_char, "floor_wood")
            char = self.char_map.get(tile_type, tile_char)
            color = self.color_map.get(tile_type, Fore.WHITE)
            return char, color
        
        return " ", Fore.WHITE
    
    def render_room(self) -> str:
        """渲染房间"""
        output = []
        output.append(f"\n{Fore.YELLOW + Style.BRIGHT}=== {self.room['name']} ==={Style.RESET_ALL}")
        output.append(f"{Fore.CYAN}{self.room.get('description', '')}{Style.RESET_ALL}\n")
        
        # 渲染地图
        for y in range(len(self.tiles)):
            row = ""
            for x in range(len(self.tiles[y])):
                char, color = self.get_char_at_position(x, y)
                row += color + char + " " + Style.RESET_ALL
            output.append(row)
        
        return "\n".join(output)
    
    def render_legend(self) -> str:
        """渲染图例"""
        output = [f"\n{Fore.YELLOW}Legend:{Style.RESET_ALL}"]
        for symbol, tile_type in self.legend.items():
            char = self.char_map.get(tile_type, symbol)
            color = self.color_map.get(tile_type, Fore.WHITE)
            output.append(f"  {color}{char}{Style.RESET_ALL} - {tile_type.replace('_', ' ').title()}")
        
        # 添加代理图例
        output.append(f"\n{Fore.YELLOW}Characters:{Style.RESET_ALL}")
        for name in self.spawn_points.keys():
            if name != "player":
                output.append(f"  {Fore.WHITE + Back.RED}{name[0].upper()}{Style.RESET_ALL} - {name}")
        
        return "\n".join(output)
    
    def render_agent_status(self, agents: Dict[str, BarAgent]) -> str:
        """渲染代理状态"""
        output = [f"\n{Fore.YELLOW}Character Status:{Style.RESET_ALL}"]
        
        for agent in agents.values():
            status = agent.get_status()
            mood_color = {
                "happy": Fore.GREEN,
                "sad": Fore.BLUE,
                "excited": Fore.YELLOW,
                "tired": Fore.MAGENTA,
                "melancholy": Fore.CYAN,
                "philosophical": Fore.WHITE,
                "neutral": Fore.WHITE
            }.get(status["mood"], Fore.WHITE)
            
            output.append(
                f"  {Fore.WHITE + Style.BRIGHT}{status['name']}{Style.RESET_ALL} "
                f"({status['role']}) - {mood_color}{status['mood']}{Style.RESET_ALL} "
                f"| Energy: {Fore.GREEN if status['energy'] > 70 else Fore.YELLOW if status['energy'] > 30 else Fore.RED}{status['energy']}%{Style.RESET_ALL}"
            )
            
            if status["recent_memories"]:
                output.append(f"    Recent: {Fore.CYAN}{status['recent_memories'][-1]}{Style.RESET_ALL}")
        
        return "\n".join(output)
    
    def render_events(self, events: List[str]) -> str:
        """渲染最近事件"""
        if not events:
            return f"\n{Fore.YELLOW}Recent Events:{Style.RESET_ALL}\n  The bar is quiet..."
        
        output = [f"\n{Fore.YELLOW}Recent Events:{Style.RESET_ALL}"]
        for event in events[-5:]:  # 显示最近5个事件
            output.append(f"  {Fore.GREEN}-{Style.RESET_ALL} {event}")
        
        return "\n".join(output)
    
    def render_menu(self) -> str:
        """渲染酒吧菜单"""
        menu = self.room.get("bar_menu", {})
        output = [f"\n{Fore.YELLOW + Style.BRIGHT}Bar Menu:{Style.RESET_ALL}"]
        
        for category, items in menu.items():
            output.append(f"  {Fore.CYAN}{category.title()}:{Style.RESET_ALL}")
            for item in items:
                output.append(f"    - {item}")
        
        return "\n".join(output)
    
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def render_full_scene(self, simulation: BarSimulation) -> str:
        """渲染完整场景"""
        self.update_agent_positions(simulation.agents)
        
        output = []
        output.append(self.render_room())
        output.append(self.render_agent_status(simulation.agents))
        output.append(self.render_events(simulation.get_recent_events()))
        output.append(self.render_legend())
        
        return "\n".join(output)

class InteractiveBarGame:
    """交互式酒吧游戏"""
    
    def __init__(self, config_path: str):
        self.renderer = BarRenderer(config_path)
        self.simulation = BarSimulation()
        self.running = True
        self.setup_game()
    
    def setup_game(self):
        """设置游戏"""
        # 创建NPC
        npc_roles = self.renderer.room["npc_roles"]
        spawn_points = self.renderer.room["spawn_points"]
        
        for name, role in npc_roles.items():
            position = spawn_points[name]
            agent = BarAgent(name, role.split(" - ")[0], position)
            self.simulation.add_agent(agent)
    
    def display_help(self):
        """显示帮助信息"""
        help_text = f"""
{Fore.YELLOW + Style.BRIGHT}Available Commands:{Style.RESET_ALL}
  {Fore.GREEN}help{Style.RESET_ALL} or {Fore.GREEN}h{Style.RESET_ALL} - Show this help
  {Fore.GREEN}look{Style.RESET_ALL} or {Fore.GREEN}l{Style.RESET_ALL} - Look around the bar
  {Fore.GREEN}status{Style.RESET_ALL} or {Fore.GREEN}s{Style.RESET_ALL} - Check character status
  {Fore.GREEN}events{Style.RESET_ALL} or {Fore.GREEN}e{Style.RESET_ALL} - Show recent events
  {Fore.GREEN}menu{Style.RESET_ALL} or {Fore.GREEN}m{Style.RESET_ALL} - Show bar menu
  {Fore.GREEN}talk <name>{Style.RESET_ALL} - Talk to a character (e.g., 'talk Bob')
  {Fore.GREEN}wait{Style.RESET_ALL} or {Fore.GREEN}w{Style.RESET_ALL} - Wait and watch (time passes)
  {Fore.GREEN}auto{Style.RESET_ALL} or {Fore.GREEN}a{Style.RESET_ALL} - Auto-simulation mode
  {Fore.GREEN}quit{Style.RESET_ALL} or {Fore.GREEN}q{Style.RESET_ALL} - Exit the game
        """
        print(help_text)
    
    def handle_talk_command(self, target_name: str):
        """处理对话命令"""
        target_name = target_name.strip().title()
        if target_name in self.simulation.agents:
            agent = self.simulation.agents[target_name]
            dialogue = agent.generate_bar_dialogue("conversation", "Player")
            print(f"\n{Fore.CYAN + Style.BRIGHT}{target_name}:{Style.RESET_ALL} \"{dialogue}\"")
            
            # 增加互动记忆
            agent.add_memory(f"I talked with the player", "social", 0.6)
        else:
            available = ", ".join(self.simulation.agents.keys())
            print(f"\n{Fore.RED}No one named '{target_name}' here. Available: {available}{Style.RESET_ALL}")
    
    def auto_simulation(self):
        """自动模拟模式"""
        print(f"\n{Fore.YELLOW}Entering auto-simulation mode. Press Ctrl+C to stop.{Style.RESET_ALL}")
        try:
            while True:
                self.renderer.clear_screen()
                print(self.renderer.render_full_scene(self.simulation))
                print(f"\n{Fore.GREEN}[Auto-simulation running... Press Ctrl+C to stop]{Style.RESET_ALL}")
                
                import time
                time.sleep(3)  # 等待3秒
                self.simulation.simulate_time_passage(5)  # 模拟5分钟
        except KeyboardInterrupt:
            print(f"\n{Fore.YELLOW}Exiting auto-simulation mode.{Style.RESET_ALL}")
    
    def run(self):
        """运行游戏"""
        print(f"{Fore.YELLOW + Style.BRIGHT}Welcome to the Cozy Bar Demo!{Style.RESET_ALL}")
        print(f"Type '{Fore.GREEN}help{Style.RESET_ALL}' for available commands.")
        
        # 初始显示
        print(self.renderer.render_full_scene(self.simulation))
        
        while self.running:
            try:
                command = input(f"\n{Fore.GREEN}> {Style.RESET_ALL}").strip().lower()
                
                if command in ['quit', 'q', 'exit']:
                    self.running = False
                    print(f"{Fore.YELLOW}Thanks for visiting the Cozy Bar! Come back soon!{Style.RESET_ALL}")
                
                elif command in ['help', 'h']:
                    self.display_help()
                
                elif command in ['look', 'l']:
                    self.renderer.clear_screen()
                    print(self.renderer.render_full_scene(self.simulation))
                
                elif command in ['status', 's']:
                    print(self.renderer.render_agent_status(self.simulation.agents))
                
                elif command in ['events', 'e']:
                    print(self.renderer.render_events(self.simulation.get_recent_events()))
                
                elif command in ['menu', 'm']:
                    print(self.renderer.render_menu())
                
                elif command in ['wait', 'w']:
                    print(f"{Fore.CYAN}Time passes...{Style.RESET_ALL}")
                    self.simulation.simulate_time_passage(10)
                    print(self.renderer.render_events(self.simulation.get_recent_events(3)))
                
                elif command in ['auto', 'a']:
                    self.auto_simulation()
                
                elif command.startswith('talk '):
                    target = command[5:]  # 移除 'talk '
                    self.handle_talk_command(target)
                
                else:
                    print(f"{Fore.RED}Unknown command. Type 'help' for available commands.{Style.RESET_ALL}")
            
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Thanks for visiting the Cozy Bar! Come back soon!{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "room_config.json")
    game = InteractiveBarGame(config_path)
    game.run()