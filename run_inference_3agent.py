import unsloth
from unsloth import FastLanguageModel
import argparse
from src.load_model import *
from src.data_preprocess import *

# 配置参数
MODEL_PATHS = {
    '0': "Qwen/Qwen2.5-3B-Instruct",
    '1': "./fliter_llm_sft",
    '2': "./model2", 
    '3': "./model3"
}

def run_single_inference(test_sample, fliter_agent, answer_agent, teacher_agent, sample_config):
    """执行单样本推理并打印结果"""
    print("\n" + "="*50)
    print(f"Processing sample ID: {test_sample['id']}")
    print("="*50)
    
    # Filter Agent阶段
    prompt_fliter = refine_for_fliter_input(test_sample)
    print("\n[Filter Agent Prompt]:\n", prompt_fliter)
    fliterd_info = get_fliter_agent_response(fliter_agent, prompt_fliter, sample_config)
    print("\n[Filter Agent Response]:\n", fliterd_info)

    # Teacher验证
    prompt_teacher = refine_for_teacher_judge_fliter(fliterd_info, test_sample)
    judgement = teacher_agent_judge(get_teacher_agent_response(teacher_agent, prompt_teacher))
    print("\n[Teacher Judgement]:", judgement)

    if judgement == "NO":
        print("\n⚠️ Retrying Filter Agent with higher temperature...")
        fliterd_info = get_fliter_agent_response(
            fliter_agent, prompt_fliter, 
            SamplingParams(temperature=1, max_tokens=1024, stop=["User:"])
        )
        print("\n[Revised Filter Response]:\n", fliterd_info)

    # Answer Agent阶段
    prompt_answer = refine_for_answer_agent(fliterd_info, test_sample['model_input']['question'])
    print("\n[Answer Agent Prompt]:\n", prompt_answer)
    final_answer = get_answer_agent_response(answer_agent, prompt_answer)
    print("\n[Answer Agent Response]:\n", final_answer)

    # Final Teacher验证
    prompt_teacher = refine_for_teacher_judge_answer(prompt_answer, final_answer)
    final_judgement = teacher_agent_judge(get_teacher_agent_response(teacher_agent, prompt_teacher))
    print("\n[Final Teacher Judgement]:", final_judgement)

    if final_judgement == "NO":
        print("\n⚠️ Retrying Answer Agent with higher temperature...")
        final_answer = get_answer_agent_response(
            answer_agent, prompt_answer,
            SamplingParams(temperature=1, max_tokens=1024, stop=["User:"])
        )
        print("\n[Revised Final Answer]:\n", final_answer)

    print("\n" + "="*50)
    print(f"Final Result for {test_sample['id']}:")
    print("="*50)
    print(final_answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='单样本推理控制台版')
    parser.add_argument('--models', type=str, required=True, 
                      help='模型组合，如123表示: filter=1, answer=2, teacher=3')

    args = parser.parse_args()

    # 参数验证
    if not (len(args.models) == 3 and all(c in '0123' for c in args.models)):
        raise ValueError("模型参数必须是3位数字，每位取值0-3")

    # 加载数据
    dataset = preprocess_hotpot_file("./hotpotqa_test_400.json")
    target_sample = dataset[2]

    # 加载模型
    print("\n🔧 Loading models...")
    fliter_agent, _, sample_config = load_fliter_agent_model(MODEL_PATHS[args.models[0]])
    answer_agent, _, _ = load_answer_agent_model(MODEL_PATHS[args.models[1]])
    teacher_agent, _, _ = load_teacher_agent_model(MODEL_PATHS[args.models[2]])

    # 执行推理
    run_single_inference(target_sample, fliter_agent, answer_agent, teacher_agent, sample_config)