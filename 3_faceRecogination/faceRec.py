import os
import cv2
import numpy as np
import pickle
import shutil
from insightface.app import FaceAnalysis

CACHE_FILE = 'embedding_cache.pkl'

# 캐시 사용 여부 설정 (디버깅 중일 땐 False로 설정)
use_cache = False


def load_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return {}


def save_cache(cache, cache_path):
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)


def imread_unicode(image_path):
    try:
        data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"이미지 디코딩 실패: {image_path}")
        return img
    except Exception as e:
        print(f"이미지 로드 실패: {image_path}, 에러: {e}")
        return None


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def extract_embedding(model, image_path):
    img = imread_unicode(image_path)
    if img is None:
        return None
    faces = model.get(img)
    if len(faces) == 0:
        print(f"⚠ 얼굴을 찾지 못함: {image_path}")
        return None
    face = max(faces, key=lambda x: x.det_score)
    emb = face.embedding
    emb = emb / np.linalg.norm(emb)
    return emb


def main():
    model_pack = 'antelopev2'
    print(f"사용할 모델 팩: {model_pack}")

    model = FaceAnalysis(name=model_pack)
    model.prepare(ctx_id=0, det_size=(640, 640))
    print("📦 로딩된 모델 모듈:", model.models.keys())

    # 타겟 임베딩 추출
    target_dir = 'data/target'
    target_files = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    target_embeddings = []
    for img_path in target_files:
        emb = extract_embedding(model, img_path)
        if emb is not None:
            target_embeddings.append(emb)
        else:
            print(f"⚠ 타겟 이미지 얼굴 인식 실패: {img_path}")

    if not target_embeddings:
        print("🚫 대상 인물의 얼굴 임베딩을 찾을 수 없습니다.")
        return

    target_embedding = np.mean(target_embeddings, axis=0)
    target_embedding /= np.linalg.norm(target_embedding)
    print(f"🎯 평균 임베딩 정규화 값: {np.linalg.norm(target_embedding):.4f}")

    # 캐시 로드
    cache = {} if not use_cache else load_cache(CACHE_FILE)

    celeb_dir = 'data/celeb'
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    threshold = 0.2

    celeb_files = [
        f for f in os.listdir(celeb_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    copied_images = []
    global_best_similarity = -1
    global_best_image = None

    for fname in celeb_files:
        img_path = os.path.join(celeb_dir, fname)
        mod_time = os.path.getmtime(img_path)

        if use_cache and fname in cache and cache[fname]['mod_time'] == mod_time:
            embeddings = cache[fname]['embeddings']
        else:
            img = imread_unicode(img_path)
            if img is None:
                print(f"이미지 로드 실패: {img_path}")
                continue
            faces = model.get(img)
            embeddings = []
            for face in faces:
                emb = face.embedding
                emb_norm = emb / np.linalg.norm(emb)
                embeddings.append(emb_norm)
            cache[fname] = {'mod_time': mod_time, 'embeddings': embeddings}
            if use_cache:
                save_cache(cache, CACHE_FILE)

        if not embeddings:
            continue

        best_similarity = max(cosine_similarity(target_embedding, emb) for emb in embeddings)

        # 디버깅용 유사도 출력
        print(f"[{fname}] 최고 유사도: {best_similarity:.4f}")

        if best_similarity > global_best_similarity:
            global_best_similarity = best_similarity
            global_best_image = fname

        if best_similarity >= threshold:
            print(f"✅ {fname}: 유사도 {best_similarity:.4f} - 복사됨")
            shutil.copy(img_path, os.path.join(output_dir, fname))
            copied_images.append(fname)

    if not copied_images and global_best_image is not None:
        print(f"⚠ 임계치 넘는 이미지 없음. 최고 유사도 {global_best_similarity:.4f} → {global_best_image} 복사")
        shutil.copy(os.path.join(celeb_dir, global_best_image), os.path.join(output_dir, global_best_image))


if __name__ == '__main__':
    main()
