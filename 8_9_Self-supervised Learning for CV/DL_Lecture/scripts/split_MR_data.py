import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_mr_dataset():
    """
    MR 데이터를 train/dev/test로 분할하는 함수
    """

    # ========================================
    # 🎯 여기에 원본 데이터 파일 경로를 입력하세요
    # ========================================
    input_file = "D:/DeepLearningStudy/8_9_Self-supervised Learning for CV/DL_Lecture/data/MR/original_data.csv"
    # 또는 다운로드한 파일명 (예: "IMDB Dataset.csv", "movie_reviews.csv" 등)

    output_dir = "D:/DeepLearningStudy/8_9_Self-supervised Learning for CV/DL_Lecture/data/MR/"

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 원본 파일 로딩: {input_file}")

    # CSV 파일 로드
    try:
        df = pd.read_csv(input_file)
        print(f"✅ 데이터 로드 성공: {len(df)}개 샘플")
        print(f"📊 컬럼들: {list(df.columns)}")
        print(f"🔍 첫 몇 개 샘플:")
        print(df.head())

    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        print("🔧 input_file 경로를 확인해주세요.")
        return
    except Exception as e:
        print(f"❌ 파일 로드 중 에러: {e}")
        return

    # 컬럼명 자동 감지 및 표준화
    print("\n🔍 컬럼명 분석 중...")

    # 텍스트 컬럼 찾기
    possible_text_columns = ['text', 'review', 'sentence', 'content', 'document', 'message']
    text_column = None

    for col in possible_text_columns:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        # 첫 번째 컬럼을 텍스트로 가정
        text_column = df.columns[0]
        print(f"⚠️ 표준 텍스트 컬럼을 찾지 못했습니다. '{text_column}'을 사용합니다.")
    else:
        print(f"✅ 텍스트 컬럼 발견: '{text_column}'")

    # 라벨 컬럼 찾기
    possible_label_columns = ['label', 'sentiment', 'target', 'class', 'y']
    label_column = None

    for col in possible_label_columns:
        if col in df.columns:
            label_column = col
            break

    if label_column is None:
        # 두 번째 컬럼을 라벨로 가정
        label_column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        print(f"⚠️ 표준 라벨 컬럼을 찾지 못했습니다. '{label_column}'을 사용합니다.")
    else:
        print(f"✅ 라벨 컬럼 발견: '{label_column}'")

    # 라벨 분포 확인
    print(f"\n📊 라벨 분포:")
    print(df[label_column].value_counts())

    # 라벨 표준화 (0/1로 변환)
    unique_labels = df[label_column].unique()
    print(f"🏷️ 고유 라벨: {unique_labels}")

    if set(unique_labels) == {0, 1}:
        print("✅ 라벨이 이미 0/1 형태입니다.")
    elif set(unique_labels) == {'positive', 'negative'}:
        print("🔄 'positive'/'negative' → 1/0 변환 중...")
        df[label_column] = df[label_column].map({'positive': 1, 'negative': 0})
    elif set(unique_labels) == {'pos', 'neg'}:
        print("🔄 'pos'/'neg' → 1/0 변환 중...")
        df[label_column] = df[label_column].map({'pos': 1, 'neg': 0})
    else:
        print(f"⚠️ 예상치 못한 라벨 형태: {unique_labels}")
        print("첫 번째 라벨을 0, 두 번째 라벨을 1로 매핑합니다.")
        label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
        df[label_column] = df[label_column].map(label_mapping)
        print(f"매핑: {label_mapping}")

    # 최종 데이터프레임 정리
    final_df = df[[text_column, label_column]].copy()
    final_df.columns = ['text', 'label']  # 표준 컬럼명으로 변경

    print(f"\n📊 최종 데이터 정보:")
    print(f"- 총 샘플 수: {len(final_df)}")
    print(f"- 라벨 분포: {final_df['label'].value_counts().to_dict()}")

    # 데이터 분할
    print(f"\n🔪 데이터 분할 중...")

    # 1단계: train + temp (80% + 20%)
    train_df, temp_df = train_test_split(
        final_df,
        test_size=0.2,
        random_state=42,
        stratify=final_df['label']  # 라벨 비율 유지
    )

    # 2단계: temp → dev + test (각각 10%)
    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label']  # 라벨 비율 유지
    )

    print(f"✅ 분할 완료:")
    print(f"  - Train: {len(train_df)}개 ({len(train_df) / len(final_df) * 100:.1f}%)")
    print(f"  - Dev:   {len(dev_df)}개 ({len(dev_df) / len(final_df) * 100:.1f}%)")
    print(f"  - Test:  {len(test_df)}개 ({len(test_df) / len(final_df) * 100:.1f}%)")

    # 각 세트의 라벨 분포 확인
    print(f"\n📊 분할된 데이터의 라벨 분포:")
    print(f"Train - 0: {(train_df['label'] == 0).sum()}, 1: {(train_df['label'] == 1).sum()}")
    print(f"Dev   - 0: {(dev_df['label'] == 0).sum()}, 1: {(dev_df['label'] == 1).sum()}")
    print(f"Test  - 0: {(test_df['label'] == 0).sum()}, 1: {(test_df['label'] == 1).sum()}")

    # CSV 파일로 저장
    train_path = os.path.join(output_dir, "mr_train.csv")
    dev_path = os.path.join(output_dir, "mr_dev.csv")
    test_path = os.path.join(output_dir, "mr_test.csv")

    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n💾 파일 저장 완료:")
    print(f"  - {train_path}")
    print(f"  - {dev_path}")
    print(f"  - {test_path}")

    # 샘플 확인
    print(f"\n🔍 저장된 파일 샘플:")
    print("Train 샘플:")
    print(train_df.head(2))
    print("\nDev 샘플:")
    print(dev_df.head(2))
    print("\nTest 샘플:")
    print(test_df.head(2))

    print(f"\n🎉 데이터 분할 완료! 이제 YAML 파일에서 경로를 확인해주세요.")


if __name__ == "__main__":
    split_mr_dataset()