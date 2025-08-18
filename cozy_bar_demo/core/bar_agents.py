"""
酒吧NPC代理系统 - 具有情感、记忆和互动的智能角色
"""
import random
import time
from typing import Tuple, List, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class Memory:
    """记忆片段"""
    content: str
    timestamp: datetime
    emotion: str
    importance: float

class SimpleAgent:
    """基础代理类"""
    def __init__(self, name: str, role: str, position: Tuple[int, int]):
        self.name = name
        self.role = role
        self.position = position
        self.memories: List[Memory] = []
        self.last_action_time = time.time()
        
    def add_memory(self, content: str, emotion: str = "neutral", importance: float = 0.5):
        """添加记忆"""
        memory = Memory(content, datetime.now(), emotion, importance)
        self.memories.append(memory)
        # 保持最近的20条记忆
        if len(self.memories) > 20:
            self.memories.pop(0)

class BarAgent(SimpleAgent):
    """酒吧NPC - 更有趣的行为"""
    
    def __init__(self, name: str, role: str, position: Tuple[int, int]):
        super().__init__(name, role, position)
        self.drunk_level = 0  # 醉酒程度 (0-10)
        self.mood = "neutral"  # 心情: happy, sad, neutral, excited, tired
        self.energy = 100  # 精力值 (0-100)
        self.social_battery = 100  # 社交电量 (0-100)
        self.favorite_drink = self._assign_favorite_drink()
        self.conversation_topics = []
        self.relationships = {}  # 与其他角色的关系
        self.last_interaction = None
        
    def _assign_favorite_drink(self) -> str:
        """根据角色分配偏好饮品"""
        drinks_by_role = {
            "bartender": ["Coffee", "Water", "Local IPA"],
            "regular customer": ["Whiskey Sour", "Old Fashioned", "Whiskey"],
            "musician": ["Beer", "Wine", "Manhattan"]
        }
        return random.choice(drinks_by_role.get(self.role, ["Water"]))
    
    def update_mood_and_energy(self):
        """更新心情和精力"""
        current_time = datetime.now()
        hour = current_time.hour
        
        # 根据时间调整精力
        if 18 <= hour <= 23:  # 酒吧营业高峰期
            self.energy = max(60, self.energy)
        elif hour < 6:  # 深夜
            self.energy = max(20, self.energy - 10)
        
        # 根据角色调整心情
        if self.role == "musician" and 19 <= hour <= 22:
            self.mood = "excited"
        elif self.role == "regular customer" and self.drunk_level > 5:
            self.mood = random.choice(["melancholy", "philosophical", "nostalgic"])
    
    def bar_specific_actions(self) -> str:
        """酒吧特定行为"""
        self.update_mood_and_energy()
        
        if self.role == "bartender":
            actions = [
                "wiping the bar counter with a clean cloth",
                "mixing a cocktail with practiced precision",
                "chatting with customers about their day",
                "checking inventory and organizing bottles",
                "polishing glasses until they gleam",
                "preparing garnishes for cocktails",
                "listening attentively to a customer's story"
            ]
        elif self.role == "regular customer":
            if self.mood == "melancholy":
                actions = [
                    "staring into the amber depths of whiskey",
                    "tracing patterns on the condensation of the glass",
                    "sighing deeply and taking a slow sip",
                    "looking out the window with distant eyes"
                ]
            elif self.mood == "philosophical":
                actions = [
                    "contemplating life over a drink",
                    "writing notes on a napkin",
                    "engaging in deep conversation with anyone who'll listen",
                    "asking existential questions to the bartender"
                ]
            else:
                actions = [
                    "sipping whiskey slowly and savoring the taste",
                    "ordering another drink with a subtle nod",
                    "telling a story from the old days",
                    "quietly observing the other patrons"
                ]
        elif self.role == "musician":
            if self.mood == "excited":
                actions = [
                    "tuning the guitar with meticulous care",
                    "practicing chord progressions",
                    "humming a new melody",
                    "preparing for the evening performance"
                ]
            else:
                actions = [
                    "taking a well-deserved break",
                    "chatting with fans about music",
                    "sipping a drink while people-watching",
                    "scribbling lyrics in a worn notebook"
                ]
        
        action = random.choice(actions)
        self.add_memory(f"I was {action}", self.mood, 0.3)
        return action
    
    def generate_bar_dialogue(self, context: str = "", target: str = None) -> str:
        """生成酒吧对话"""
        current_hour = datetime.now().hour
        
        if self.role == "bartender":
            if "order" in context.lower():
                responses = [
                    "What can I get you tonight?",
                    "The usual?",
                    f"Might I suggest a {random.choice(['Old Fashioned', 'Manhattan', 'Whiskey Sour'])}?",
                    "What mood are you drinking for tonight?"
                ]
            elif self.mood == "friendly":
                responses = [
                    "Rough day, huh? Let me fix that for you.",
                    "This one's on the house - you look like you need it.",
                    "I've been mixing drinks for 20 years, and let me tell you...",
                    "What's your story? Everyone who sits here has one."
                ]
            else:
                responses = [
                    "Evening. What brings you to our corner of the world?",
                    "Take your time deciding - good drinks can't be rushed.",
                    "We've got some excellent local spirits if you're interested.",
                    "How's the night treating you so far?"
                ]
                
        elif self.role == "regular customer":
            if self.mood == "melancholy":
                responses = [
                    "Leave me alone with my drink... and my thoughts.",
                    "You know what? Life's funny... not ha-ha funny, but strange funny.",
                    "Sometimes I wonder what would've happened if...",
                    "This glass holds more than whiskey - it holds memories."
                ]
            elif self.mood == "philosophical":
                responses = [
                    "Ever notice how this place becomes a confessional after 9 PM?",
                    "What defines a person? Their actions or their intentions?",
                    "Bob! Another round - and bring your wisdom with it!",
                    "Time moves differently in places like this, don't you think?"
                ]
            elif self.drunk_level > 6:
                responses = [
                    "Let me tell you 'bout the time I... wait, what was I saying?",
                    "You're my best friend, you know that? Best. Friend.",
                    "I love this song! Sam, play it again!",
                    "Life's too short for bad whiskey and long grudges."
                ]
            else:
                responses = [
                    "Another round, Bob - and make it a double this time.",
                    "Ever feel like nothing matters? Then everything matters?",
                    "I've been coming here for... how long has it been, Bob?",
                    "Cheers to the stories we'll never tell."
                ]
                
        elif self.role == "musician":
            if current_hour >= 20 and self.mood == "excited":
                responses = [
                    "Any requests for tonight's set?",
                    "This next song goes out to everyone who's ever loved and lost.",
                    "Music helps, doesn't it? It says what we can't.",
                    "I'm playing at 9 sharp - stick around for something special!"
                ]
            elif self.energy < 50:
                responses = [
                    "Just finished a set - time for a well-earned break.",
                    "My fingers are tired but my heart's still singing.",
                    "Sometimes the music plays you, not the other way around.",
                    "There's nothing like a cold beer after a hot performance."
                ]
            else:
                responses = [
                    "Preparing for tonight's performance - got any favorites?",
                    "The acoustics in here are perfect for intimate songs.",
                    "Been working on a new piece - melancholy with a hint of hope.",
                    "Music is the universal language, don't you think?"
                ]
        
        response = random.choice(responses)
        self.add_memory(f"I said: '{response}' to {target or 'someone'}", self.mood, 0.4)
        return response
    
    def interact_with(self, other_agent: 'BarAgent') -> str:
        """与其他角色互动"""
        if other_agent.name not in self.relationships:
            self.relationships[other_agent.name] = 0.0  # 中性关系
        
        # 更新关系值
        interaction_quality = random.uniform(-0.1, 0.2)
        self.relationships[other_agent.name] += interaction_quality
        
        # 生成互动内容
        if self.role == "bartender" and other_agent.role == "regular customer":
            interactions = [
                f"serves {other_agent.name} a drink with a knowing nod",
                f"shares a quiet moment of understanding with {other_agent.name}",
                f"listens patiently as {other_agent.name} shares their thoughts"
            ]
        elif self.role == "musician" and other_agent.role == "regular customer":
            interactions = [
                f"dedicates a song to {other_agent.name}",
                f"shares music stories with {other_agent.name}",
                f"invites {other_agent.name} to request a song"
            ]
        else:
            interactions = [
                f"has a casual conversation with {other_agent.name}",
                f"shares a laugh with {other_agent.name}",
                f"enjoys a moment of comfortable silence with {other_agent.name}"
            ]
        
        interaction = random.choice(interactions)
        self.add_memory(f"I {interaction}", "social", 0.5)
        self.last_interaction = datetime.now()
        return interaction
    
    def get_status(self) -> Dict:
        """获取角色状态"""
        return {
            "name": self.name,
            "role": self.role,
            "position": self.position,
            "mood": self.mood,
            "energy": self.energy,
            "drunk_level": self.drunk_level,
            "social_battery": self.social_battery,
            "favorite_drink": self.favorite_drink,
            "recent_memories": [m.content for m in self.memories[-3:]]
        }

class BarSimulation:
    """酒吧模拟系统"""
    
    def __init__(self):
        self.agents = {}
        self.time_factor = 60  # 1分钟 = 1小时游戏时间
        self.events = []
        
    def add_agent(self, agent: BarAgent):
        """添加代理"""
        self.agents[agent.name] = agent
        
    def simulate_time_passage(self, minutes: int = 1):
        """模拟时间流逝"""
        for agent in self.agents.values():
            # 随机事件
            if random.random() < 0.3:  # 30%概率发生事件
                action = agent.bar_specific_actions()
                self.events.append(f"{agent.name} is {action}")
            
            # 角色互动
            if random.random() < 0.2 and len(self.agents) > 1:  # 20%概率互动
                other_agents = [a for a in self.agents.values() if a != agent]
                other = random.choice(other_agents)
                interaction = agent.interact_with(other)
                self.events.append(f"{agent.name} {interaction}")
    
    def get_scene_description(self) -> str:
        """获取场景描述"""
        descriptions = []
        for agent in self.agents.values():
            status = agent.get_status()
            descriptions.append(
                f"{agent.name} ({agent.role}) is at position {agent.position}, "
                f"feeling {agent.mood}, with {agent.energy}% energy"
            )
        return "\n".join(descriptions)
    
    def get_recent_events(self, count: int = 5) -> List[str]:
        """获取最近事件"""
        return self.events[-count:] if self.events else ["The bar is quiet..."]