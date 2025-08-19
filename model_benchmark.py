"""
模型性能基准测试工具
比较不同LLM模型在生成任务中的性能表现，包括：
- 响应时间
- 响应质量 
- 内存使用
- 成功率
"""

import time
import psutil
import json
import asyncio
import statistics
import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# 添加ai_service路径
sys.path.append(str(Path(__file__).parent / "ai_service"))

try:
    from ai_service.config import get_config
    from ai_service.local_llm_adapter import safe_generate_response
    from ai_service.ai_service import local_llm_generate, set_active_model, get_active_model
except ImportError as e:
    print(f"❌ 无法导入AI服务模块: {e}")
    print("请确保AI服务正在运行或者模块路径正确")
    sys.exit(1)

@dataclass
class BenchmarkResult:
    """单个测试结果"""
    model_name: str
    prompt: str
    response: str
    response_time_ms: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None
    tokens_per_second: Optional[float] = None
    response_length: int = 0

@dataclass 
class ModelBenchmarkSummary:
    """模型基准测试汇总"""
    model_name: str
    total_tests: int
    successful_tests: int
    failed_tests: int
    success_rate: float
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    std_response_time_ms: float
    avg_memory_usage_mb: float
    avg_tokens_per_second: float
    avg_response_length: float
    quality_score: float = 0.0

class ModelBenchmark:
    """模型性能基准测试器"""
    
    def __init__(self):
        """初始化基准测试器"""
        self.config = get_config()
        self.model_config = self.config.get_model_config()
        self.supported_models = self.model_config.supported_models
        self.results: List[BenchmarkResult] = []
        self.test_prompts = self._get_test_prompts()
        
    def _get_test_prompts(self) -> List[Dict[str, str]]:
        """获取测试提示词集合"""
        return [
            {
                "name": "简单对话",
                "prompt": "你好，请简单介绍一下自己。",
                "category": "conversation"
            },
            {
                "name": "创意写作",
                "prompt": "写一个关于咖啡店的50字小故事。",
                "category": "creative_writing"
            },
            {
                "name": "技术解释",
                "prompt": "用简单的语言解释什么是机器学习。",
                "category": "explanation"
            },
            {
                "name": "问题解决",
                "prompt": "如何提高工作效率？请给出3个具体建议。",
                "category": "problem_solving"
            },
            {
                "name": "角色扮演",
                "prompt": "假设你是一个咖啡店老板，一位顾客抱怨咖啡太苦，你会如何回应？",
                "category": "role_play"
            },
            {
                "name": "逻辑推理",
                "prompt": "如果所有的鸟都会飞，企鹅是鸟，那么企鹅会飞吗？请解释为什么。",
                "category": "logic"
            },
            {
                "name": "JSON格式",
                "prompt": "请以JSON格式描述一个人的基本信息，包含姓名、年龄、职业。",
                "category": "structured_output"
            },
            {
                "name": "情感分析",
                "prompt": "分析这句话的情感色彩：'今天的天气真是糟糕透了，又下雨又冷。'",
                "category": "analysis"
            }
        ]
    
    def _get_memory_usage_mb(self) -> float:
        """获取当前内存使用量(MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _estimate_tokens_per_second(self, text: str, time_ms: float) -> float:
        """估算token/秒 (粗略估计：1个中文字符≈1个token，英文单词≈1个token)"""
        if time_ms <= 0:
            return 0.0
        
        # 简单估计token数
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_words = len([w for w in text.split() if w.isalnum() and not any('\u4e00' <= c <= '\u9fff' for c in w)])
        estimated_tokens = chinese_chars + english_words
        
        return estimated_tokens / (time_ms / 1000)
    
    def _evaluate_response_quality(self, prompt: str, response: str, category: str) -> float:
        """评估响应质量 (0-100分)"""
        score = 50  # 基础分
        
        if not response or len(response.strip()) < 10:
            return 20  # 响应太短或为空
        
        # 长度合理性 (20分)
        if 20 <= len(response) <= 500:
            score += 20
        elif len(response) > 500:
            score += 15
        elif len(response) >= 10:
            score += 10
        
        # 相关性检查 (30分) - 简单关键词匹配
        prompt_keywords = set(prompt.lower().split())
        response_keywords = set(response.lower().split())
        relevance = len(prompt_keywords.intersection(response_keywords)) / max(len(prompt_keywords), 1)
        score += relevance * 30
        
        # 根据类别特殊评分
        if category == "structured_output" and ("{" in response and "}" in response):
            score += 10  # JSON格式奖励
        elif category == "creative_writing" and len(response) >= 40:
            score += 10  # 创意写作长度奖励
        elif category == "explanation" and ("是" in response or "因为" in response or "所以" in response):
            score += 10  # 解释性内容奖励
        
        return min(score, 100)
    
    async def test_model(self, model_name: str, prompt_data: Dict[str, str]) -> BenchmarkResult:
        """测试单个模型的单个提示词"""
        print(f"  📝 测试: {prompt_data['name']}")
        
        # 获取测试前内存使用
        memory_before = self._get_memory_usage_mb()
        
        # 设置活动模型
        original_model = get_active_model()
        if model_name != original_model:
            set_active_model(model_name)
        
        start_time = time.time()
        success = True
        response = ""
        error_message = None
        
        try:
            # 生成响应
            response = local_llm_generate(
                prompt_data["prompt"], 
                model_key=model_name,
                max_retries=1
            )
            
            if not response:
                success = False
                error_message = "空响应"
                
        except Exception as e:
            success = False
            error_message = str(e)
            response = ""
        
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        # 获取测试后内存使用
        memory_after = self._get_memory_usage_mb()
        memory_usage = memory_after - memory_before
        
        # 计算token/秒
        tokens_per_second = self._estimate_tokens_per_second(response, response_time_ms) if success else 0
        
        # 恢复原模型
        if original_model != model_name:
            set_active_model(original_model)
        
        result = BenchmarkResult(
            model_name=model_name,
            prompt=prompt_data["prompt"],
            response=response,
            response_time_ms=response_time_ms,
            memory_usage_mb=memory_usage,
            success=success,
            error_message=error_message,
            tokens_per_second=tokens_per_second,
            response_length=len(response)
        )
        
        self.results.append(result)
        return result
    
    def _calculate_model_summary(self, model_name: str) -> ModelBenchmarkSummary:
        """计算模型的统计汇总"""
        model_results = [r for r in self.results if r.model_name == model_name]
        
        if not model_results:
            return ModelBenchmarkSummary(
                model_name=model_name,
                total_tests=0,
                successful_tests=0,
                failed_tests=0,
                success_rate=0.0,
                avg_response_time_ms=0.0,
                min_response_time_ms=0.0,
                max_response_time_ms=0.0,
                std_response_time_ms=0.0,
                avg_memory_usage_mb=0.0,
                avg_tokens_per_second=0.0,
                avg_response_length=0.0
            )
        
        successful_results = [r for r in model_results if r.success]
        failed_results = [r for r in model_results if not r.success]
        
        # 计算响应时间统计
        response_times = [r.response_time_ms for r in successful_results] if successful_results else [0]
        
        # 计算质量分数
        quality_scores = []
        for i, result in enumerate(model_results):
            if result.success and i < len(self.test_prompts):
                quality = self._evaluate_response_quality(
                    result.prompt, 
                    result.response, 
                    self.test_prompts[i]["category"]
                )
                quality_scores.append(quality)
        
        avg_quality = statistics.mean(quality_scores) if quality_scores else 0
        
        return ModelBenchmarkSummary(
            model_name=model_name,
            total_tests=len(model_results),
            successful_tests=len(successful_results),
            failed_tests=len(failed_results),
            success_rate=len(successful_results) / len(model_results) * 100,
            avg_response_time_ms=statistics.mean(response_times),
            min_response_time_ms=min(response_times),
            max_response_time_ms=max(response_times),
            std_response_time_ms=statistics.stdev(response_times) if len(response_times) > 1 else 0,
            avg_memory_usage_mb=statistics.mean([r.memory_usage_mb for r in model_results]),
            avg_tokens_per_second=statistics.mean([r.tokens_per_second for r in successful_results]) if successful_results else 0,
            avg_response_length=statistics.mean([r.response_length for r in successful_results]) if successful_results else 0,
            quality_score=avg_quality
        )
    
    async def run_benchmark(self, models: Optional[List[str]] = None) -> Dict[str, ModelBenchmarkSummary]:
        """运行完整基准测试"""
        if models is None:
            models = list(self.supported_models.keys())
        
        print("🚀 开始模型基准测试...")
        print(f"📊 测试模型: {', '.join(models)}")
        print(f"📝 测试用例: {len(self.test_prompts)}个")
        print("-" * 60)
        
        summaries = {}
        
        for model_name in models:
            if model_name not in self.supported_models:
                print(f"⚠️  警告: 模型 '{model_name}' 不在支持列表中，跳过")
                continue
                
            print(f"\n🔍 测试模型: {model_name}")
            print(f"📁 模型文件: {self.supported_models[model_name]}")
            
            # 测试每个提示词
            for prompt_data in self.test_prompts:
                try:
                    result = await self.test_model(model_name, prompt_data)
                    status = "✅ 成功" if result.success else f"❌ 失败: {result.error_message}"
                    print(f"    {status} ({result.response_time_ms:.0f}ms)")
                    
                    # 短暂延迟避免过载
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    print(f"    ❌ 异常: {str(e)}")
                    continue
            
            # 计算模型汇总
            summaries[model_name] = self._calculate_model_summary(model_name)
        
        return summaries
    
    def print_results(self, summaries: Dict[str, ModelBenchmarkSummary]):
        """打印测试结果"""
        print("\n" + "="*80)
        print("📊 模型性能基准测试结果")
        print("="*80)
        
        # 按综合评分排序
        sorted_models = sorted(
            summaries.items(), 
            key=lambda x: (x[1].success_rate * 0.4 + (100 - x[1].avg_response_time_ms/100) * 0.3 + x[1].quality_score * 0.3),
            reverse=True
        )
        
        for i, (model_name, summary) in enumerate(sorted_models, 1):
            print(f"\n🏆 #{i} {model_name.upper()}")
            print("-" * 40)
            print(f"成功率:     {summary.success_rate:.1f}% ({summary.successful_tests}/{summary.total_tests})")
            print(f"平均响应:   {summary.avg_response_time_ms:.0f}ms")
            print(f"响应范围:   {summary.min_response_time_ms:.0f}-{summary.max_response_time_ms:.0f}ms")
            print(f"质量评分:   {summary.quality_score:.1f}/100")
            print(f"生成速度:   {summary.avg_tokens_per_second:.1f} tokens/秒")
            print(f"平均长度:   {summary.avg_response_length:.0f}字符")
            print(f"内存开销:   {summary.avg_memory_usage_mb:.1f}MB")
            
            # 性能等级
            if summary.avg_response_time_ms < 2000:
                performance_level = "🚀 极快"
            elif summary.avg_response_time_ms < 5000:
                performance_level = "⚡ 快速"
            elif summary.avg_response_time_ms < 10000:
                performance_level = "🐌 中等"
            else:
                performance_level = "🐢 较慢"
            
            print(f"性能等级:   {performance_level}")
        
        # 推荐使用场景
        print(f"\n💡 使用建议:")
        if len(sorted_models) >= 2:
            fastest = min(summaries.items(), key=lambda x: x[1].avg_response_time_ms)
            highest_quality = max(summaries.items(), key=lambda x: x[1].quality_score)
            most_reliable = max(summaries.items(), key=lambda x: x[1].success_rate)
            
            print(f"🚀 最快响应: {fastest[0]} ({fastest[1].avg_response_time_ms:.0f}ms)")
            print(f"🎯 最高质量: {highest_quality[0]} ({highest_quality[1].quality_score:.1f}分)")
            print(f"🛡️ 最可靠:   {most_reliable[0]} ({most_reliable[1].success_rate:.1f}%)")
    
    def save_results(self, summaries: Dict[str, ModelBenchmarkSummary], filename: Optional[str] = None):
        """保存测试结果到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_benchmark_{timestamp}.json"
        
        # 准备数据
        data = {
            "timestamp": datetime.now().isoformat(),
            "test_config": {
                "test_prompts": len(self.test_prompts),
                "supported_models": self.supported_models
            },
            "summaries": {name: asdict(summary) for name, summary in summaries.items()},
            "detailed_results": [asdict(result) for result in self.results]
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"\n💾 结果已保存到: {filename}")
        except Exception as e:
            print(f"\n❌ 保存失败: {e}")

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM模型性能基准测试")
    parser.add_argument("--models", "-m", nargs="+", help="指定要测试的模型 (默认测试所有)")
    parser.add_argument("--output", "-o", help="输出文件名")
    parser.add_argument("--quick", "-q", action="store_true", help="快速测试（只测试部分用例）")
    
    args = parser.parse_args()
    
    # 创建基准测试器
    benchmark = ModelBenchmark()
    
    # 快速模式只测试前4个用例
    if args.quick:
        benchmark.test_prompts = benchmark.test_prompts[:4]
        print("⚡ 快速测试模式：使用精简测试用例")
    
    try:
        # 运行基准测试
        summaries = await benchmark.run_benchmark(args.models)
        
        if not summaries:
            print("❌ 没有成功测试任何模型")
            return
        
        # 显示结果
        benchmark.print_results(summaries)
        
        # 保存结果
        benchmark.save_results(summaries, args.output)
        
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出现异常: {e}")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())