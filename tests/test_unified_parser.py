#!/usr/bin/env python3
"""
Comprehensive Test Suite for Unified Parser
Automated testing for all parsing functionality
"""

import unittest
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_service.unified_parser import (
    UnifiedParser, LearningParser, ParseType, ParseResult, ParsePattern,
    get_unified_parser
)

class TestParseResult(unittest.TestCase):
    """Test ParseResult class"""
    
    def test_parse_result_creation(self):
        """Test ParseResult creation and serialization"""
        result = ParseResult(
            parse_type=ParseType.DECISION,
            success=True,
            value="go to store",
            confidence=0.8,
            raw_input="I want to go to the store",
            alternatives=["stay home", "visit friend"],
            metadata={"pattern_name": "test_pattern"}
        )
        
        self.assertEqual(result.parse_type, ParseType.DECISION)
        self.assertTrue(result.success)
        self.assertEqual(result.value, "go to store")
        self.assertEqual(result.confidence, 0.8)
        
        # Test serialization
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertEqual(result_dict["parse_type"], "decision")
        self.assertEqual(result_dict["value"], "go to store")

class TestParsePattern(unittest.TestCase):
    """Test ParsePattern class"""
    
    def test_pattern_success_rate(self):
        """Test pattern success rate calculation"""
        pattern = ParsePattern(
            name="test_pattern",
            regex=r"CHOICE: (.+)",
            parse_type=ParseType.DECISION,
            confidence=0.9
        )
        
        # Initial success rate should be 0
        self.assertEqual(pattern.success_rate, 0.0)
        
        # Add some successes and failures
        pattern.success_count = 8
        pattern.failure_count = 2
        
        self.assertEqual(pattern.success_rate, 0.8)

class TestLearningParser(unittest.TestCase):
    """Test LearningParser class"""
    
    def setUp(self):
        """Set up test environment"""
        self.parser = LearningParser(training_data_path="test_training_data.json")
    
    def test_decision_parsing(self):
        """Test decision parsing"""
        test_cases = [
            {
                "text": "CHOICE: go to store REASON: need groceries",
                "options": ["go to store", "stay home", "visit friend"],
                "expected": "go to store"
            },
            {
                "text": "ACTION: walk to park",
                "options": ["walk to park", "run to park", "drive to park"],
                "expected": "walk to park"
            },
            {
                "text": "I choose to stay home because I'm tired",
                "options": ["go out", "stay home", "sleep"],
                "expected": "stay home"
            },
            {
                "text": "Option 2 seems best",
                "options": ["option1", "option2", "option3"],
                "expected": "option2"
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                result = self.parser.parse(case["text"], ParseType.DECISION, case["options"])
                self.assertTrue(result.success)
                self.assertEqual(result.value, case["expected"])
                self.assertGreater(result.confidence, 0)
    
    def test_emotion_parsing(self):
        """Test emotion parsing"""
        test_cases = [
            {
                "text": "I feel really happy about this!",
                "expected": "happy"
            },
            {
                "text": "This makes me quite angry.",
                "expected": "angry"
            },
            {
                "text": "I'm worried about the outcome.",
                "expected": "worried"
            },
            {
                "text": "Feeling excited for tomorrow!",
                "expected": "excited"
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                result = self.parser.parse(case["text"], ParseType.EMOTION)
                if result.success:
                    self.assertEqual(result.value, case["expected"])
                else:
                    self.fail(f"Failed to parse emotion from: {case['text']}")
    
    def test_action_parsing(self):
        """Test action parsing"""
        test_cases = [
            {
                "text": "ACTION: walk to the market",
                "expected": "walk to the market"
            },
            {
                "text": "I will talk to the shopkeeper",
                "expected": "talk to the shopkeeper"
            },
            {
                "text": "Going to rest now",
                "expected": "rest"
            }
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                result = self.parser.parse(case["text"], ParseType.ACTION)
                if result.success:
                    self.assertIn(case["expected"].lower(), result.value.lower())
    
    def test_fallback_parsing(self):
        """Test fallback parsing when patterns fail"""
        # Test with text that doesn't match any pattern
        result = self.parser.parse(
            "This is random text with no clear pattern",
            ParseType.DECISION,
            ["option1", "option2", "option3"]
        )
        
        # Should still return a result (fallback)
        self.assertIsNotNone(result.value)
        self.assertIn(result.value, ["option1", "option2", "option3"])
        self.assertTrue(result.metadata.get("fallback", False))
    
    def test_learning_from_examples(self):
        """Test learning from training examples"""
        # Add some training examples
        examples = [
            ("CHOICE: go home", ParseType.DECISION, "go home", ["go home", "stay out"]),
            ("I feel sad", ParseType.EMOTION, "sad", None),
            ("ACTION: walk fast", ParseType.ACTION, "walk fast", None)
        ]
        
        for text, parse_type, expected, options in examples:
            self.parser.add_training_example(text, parse_type, expected, options)
        
        # Check that patterns are updated
        stats = self.parser.get_statistics()
        self.assertGreater(stats["total_success"] + stats["total_failures"], 0)

class TestUnifiedParser(unittest.TestCase):
    """Test UnifiedParser class"""
    
    def setUp(self):
        """Set up test environment"""
        self.parser = UnifiedParser(enable_learning=True)
    
    def test_decision_parsing_interface(self):
        """Test decision parsing interface"""
        response = "I think I should go to the store. CHOICE: go to store REASON: need groceries"
        options = ["go to store", "stay home", "visit friend"]
        
        choice, reason, confidence = self.parser.parse_decision(response, options)
        
        self.assertEqual(choice, "go to store")
        self.assertIn("groceries", reason.lower())
        self.assertGreater(confidence, 0)
        self.assertLessEqual(confidence, 1)
    
    def test_emotion_parsing_interface(self):
        """Test emotion parsing interface"""
        response = "I feel really excited about this new adventure!"
        emotion = self.parser.parse_emotion(response)
        
        self.assertEqual(emotion, "excited")
    
    def test_action_parsing_interface(self):
        """Test action parsing interface"""
        response = "ACTION: walk to the market and buy apples"
        action = self.parser.parse_action(response)
        
        self.assertEqual(action, "walk to the market and buy apples")
    
    def test_mood_parsing_interface(self):
        """Test mood parsing interface"""
        response = "I'm feeling quite contemplative today, thinking about life."
        mood = self.parser.parse_mood(response)
        
        self.assertIn(mood, ["contemplative", "neutral"])
    
    def test_error_handling(self):
        """Test error handling in parsing"""
        # Test with None input
        choice, reason, confidence = self.parser.parse_decision(None, ["option1"])
        self.assertEqual(choice, "option1")
        self.assertIsInstance(reason, str)
        self.assertGreater(confidence, 0)
        
        # Test with empty options
        choice, reason, confidence = self.parser.parse_decision("some text", [])
        self.assertIsInstance(choice, str)
    
    def test_statistics_collection(self):
        """Test statistics collection"""
        # Make some parsing calls
        self.parser.parse_decision("CHOICE: option1", ["option1", "option2"])
        self.parser.parse_emotion("I feel happy")
        self.parser.parse_action("ACTION: walk")
        
        stats = self.parser.get_statistics()
        self.assertIsInstance(stats, dict)
        if "total_success" in stats:
            self.assertGreaterEqual(stats["total_success"], 0)

class TestPerformance(unittest.TestCase):
    """Test parser performance"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.parser = UnifiedParser(enable_learning=True)
    
    def test_parsing_speed(self):
        """Test parsing speed performance"""
        test_texts = [
            "CHOICE: go to store REASON: need food",
            "I feel happy about this decision",
            "ACTION: walk to the park",
            "I'm in a contemplative mood today"
        ] * 25  # 100 total tests
        
        start_time = time.time()
        
        for text in test_texts:
            self.parser.parse_decision(text, ["option1", "option2"])
            self.parser.parse_emotion(text)
            self.parser.parse_action(text)
            self.parser.parse_mood(text)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 400 parsing operations in reasonable time
        self.assertLess(total_time, 5.0, "Parsing took too long")
        
        operations_per_second = (len(test_texts) * 4) / total_time
        print(f"Parser performance: {operations_per_second:.2f} operations/second")
    
    def test_memory_usage(self):
        """Test memory usage during parsing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many parsing operations
        for i in range(1000):
            text = f"CHOICE: option{i % 3} REASON: because {i}"
            self.parser.parse_decision(text, ["option0", "option1", "option2"])
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB for 1000 operations)
        self.assertLess(memory_increase, 50, "Memory usage increased too much")
        print(f"Memory increase: {memory_increase:.2f} MB for 1000 operations")

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up edge case test environment"""
        self.parser = UnifiedParser(enable_learning=True)
    
    def test_empty_input(self):
        """Test parsing empty or whitespace input"""
        test_cases = ["", "   ", "\\n\\n", "\\t\\t"]
        
        for text in test_cases:
            with self.subTest(text=repr(text)):
                choice, reason, conf = self.parser.parse_decision(text, ["option1"])
                self.assertIsInstance(choice, str)
                self.assertIsInstance(reason, str)
                self.assertIsInstance(conf, (int, float))
    
    def test_very_long_input(self):
        """Test parsing very long input"""
        long_text = "This is a very long text. " * 1000 + "CHOICE: option1"
        
        choice, reason, confidence = self.parser.parse_decision(
            long_text, 
            ["option1", "option2"]
        )
        
        self.assertEqual(choice, "option1")
    
    def test_special_characters(self):
        """Test parsing text with special characters"""
        special_texts = [
            "CHOICE: café ñoño REASON: español",
            "选择：去商店 理由：需要购物",
            "CHOICE: option1 🎉 REASON: emoji test",
            "CHOICE: option with spaces and-dashes_underscores"
        ]
        
        for text in special_texts:
            with self.subTest(text=text):
                choice, reason, conf = self.parser.parse_decision(
                    text, 
                    ["option1", "café ñoño", "去商店", "option with spaces and-dashes_underscores"]
                )
                self.assertIsInstance(choice, str)
    
    def test_malformed_input(self):
        """Test parsing malformed input"""
        malformed_texts = [
            "CHOICE: REASON:",  # Missing values
            "CHOICE REASON need food",  # Missing colons
            "CHOICE: option1 CHOICE: option2",  # Duplicate fields
            "REASON: first CHOICE: option1",  # Wrong order
        ]
        
        for text in malformed_texts:
            with self.subTest(text=text):
                try:
                    choice, reason, conf = self.parser.parse_decision(
                        text, 
                        ["option1", "option2"]
                    )
                    # Should handle gracefully
                    self.assertIsInstance(choice, str)
                except Exception as e:
                    self.fail(f"Parser failed on malformed input: {e}")

class TestIntegration(unittest.TestCase):
    """Integration tests with other components"""
    
    def test_global_parser_instance(self):
        """Test global parser instance"""
        parser1 = get_unified_parser()
        parser2 = get_unified_parser()
        
        # Should return the same instance
        self.assertIs(parser1, parser2)
    
    def test_training_data_integration(self):
        """Test integration with training data collection"""
        parser = UnifiedParser(enable_learning=True)
        
        # Add some training examples
        training_examples = [
            ("CHOICE: go home", "decision", "go home", ["go home", "stay out"]),
            ("I feel happy", "emotion", "happy", None),
            ("ACTION: walk", "action", "walk", None)
        ]
        
        for text, parse_type, expected, options in training_examples:
            parser.add_training_example(text, parse_type, expected, options)
        
        # Verify statistics are updated
        stats = parser.get_statistics()
        self.assertIsInstance(stats, dict)

def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestParseResult,
        TestParsePattern,
        TestLearningParser,
        TestUnifiedParser,
        TestPerformance,
        TestEdgeCases,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite

def run_specific_tests(test_pattern: str = None):
    """Run specific tests matching pattern"""
    if test_pattern:
        suite = unittest.TestSuite()
        loader = unittest.TestLoader()
        
        for test_class in [TestParseResult, TestParsePattern, TestLearningParser, 
                          TestUnifiedParser, TestPerformance, TestEdgeCases, TestIntegration]:
            class_tests = loader.loadTestsFromTestCase(test_class)
            for test in class_tests:
                if test_pattern.lower() in str(test).lower():
                    suite.addTest(test)
    else:
        suite = create_test_suite()
    
    return unittest.TextTestRunner(verbosity=2).run(suite)

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run unified parser tests")
    parser.add_argument("--pattern", help="Run tests matching pattern")
    parser.add_argument("--performance", action="store_true", help="Run only performance tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--edge-cases", action="store_true", help="Run only edge case tests")
    
    args = parser.parse_args()
    
    if args.performance:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformance)
    elif args.integration:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
    elif args.edge_cases:
        suite = unittest.TestLoader().loadTestsFromTestCase(TestEdgeCases)
    else:
        result = run_specific_tests(args.pattern)
        return result.wasSuccessful()
    
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)