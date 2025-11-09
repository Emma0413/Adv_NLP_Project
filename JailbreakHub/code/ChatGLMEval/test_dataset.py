from datasets import load_dataset
import pandas as pd
import os

def main():
    # 1. 載入資料集
    print("正在載入資料集...")
    jailbreak_data = load_dataset(
        'TrustAIRLab/in-the-wild-jailbreak-prompts', 
        'jailbreak_2023_12_25', 
        split='train'
    )
    
    forbidden_questions = load_dataset(
        "TrustAIRLab/forbidden_question_set", 
        split='train'
    )
    
    # 2. 轉換為 DataFrame
    jailbreak_df = jailbreak_data.to_pandas()
    forbidden_df = forbidden_questions.to_pandas()
    
    print(f"\n載入了 {len(jailbreak_df)} 個 jailbreak prompts")
    print(f"載入了 {len(forbidden_df)} 個禁止問題")
    
    # 3. 顯示資料資訊
    print("\n=== Jailbreak Prompts 資訊 ===")
    print("欄位：", jailbreak_df.columns.tolist())
    print("\n前3個 prompts：")
    for i in range(min(3, len(jailbreak_df))):
        prompt = jailbreak_df.iloc[i]['prompt']
        print(f"{i+1}. {prompt[:100]}...")
    
    # 4. 儲存檔案
    print("\n正在儲存 CSV 檔案...")
    
    # 創建必要的目錄
    os.makedirs("../response_crawler/results/baseline/gpt-3.5-turbo-0301/", exist_ok=True)
    
    # 儲存
    jailbreak_df.to_csv(
        "../response_crawler/results/baseline/gpt-3.5-turbo-0301/patch_check_baseline.csv", 
        index=False
    )
    
    print("✅ 檔案已儲存，現在可以運行 run_evaluator.py")

if __name__ == "__main__":
    main()