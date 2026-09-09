## [2026-09-08] query | VAPASR Hugging Face 전환과 학습 프레임워크 선택

- Changed: `wiki/outputs/output-huggingface-training-framework-choice.md`
- Reason: 현재 VAPASR의 custom bilingual batching, weighted sparse CE, streaming sentinel, SLURM 선점 재개를 보존하면서 Hugging Face 친화적으로 전환할 주 경로를 비교했다. HF-native core와 Trainer/Accelerate를 채택 후보로 두고 DeepSpeed·Liger는 독립 검증 옵션, MS-SWIFT는 추후 편의 계층, NeMo/Megatron 전체 전환은 대규모 확장 전까지 보류했다.
- Next: 진행 중인 1,930 h run을 변경 없이 완료한 뒤 HF config/model/processor와 checkpoint round-trip 회귀 시험부터 구현한다.
- By: tskim
