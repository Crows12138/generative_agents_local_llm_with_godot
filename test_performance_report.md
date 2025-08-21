# Performance Analysis Report

**Generated**: 2025-08-21T15:22:58.240720

## Summary

- **Total Cycles Analyzed**: 3
- **Biggest Bottleneck**: llm_call

## Phase Analysis

| Phase | Avg (ms) | Max (ms) | Min (ms) | Std Dev | Samples | Bottleneck |
|-------|----------|----------|----------|---------|---------|------------|
| perception | 1.5 | 1.6 | 1.2 | 0.2 | 3 | OK: No |
| llm_call | 100.3 | 100.5 | 100.2 | 0.1 | 3 | OK: No |
| action_parsing | 2.3 | 2.5 | 2.1 | 0.2 | 3 | OK: No |
| execution | 5.2 | 5.3 | 5.0 | 0.1 | 3 | OK: No |

## Recommendations

### LLM Call Optimization
- Consider using smaller/faster models
- Implement response caching
- Use parallel processing for multiple agents
- Optimize prompt length

