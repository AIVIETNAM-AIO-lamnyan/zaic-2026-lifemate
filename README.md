# Zalo AI Challenge: LifeMate

## Mục lục

- [1. Tổng quan](#1-tổng-quan)
  - [1.1 Pipeline](#11-pipeline)
  - [1.2 Tech Stack](#12-tech-stack)
- [2. Cấu trúc dự án và dữ liệu](#2-cấu-trúc-dự-án-và-dữ-liệu)
- [3. Chuẩn bị môi trường](#3-chuẩn-bị-môi-trường)
- [4. Cấu hình](#4-cấu-hình)
- [5. Khởi động vLLM](#5-khởi-động-vllm)
- [6. Chạy dự đoán](#6-chạy-dự-đoán)
  - [6.1 Public test](#61-public-test)
  - [6.2 Train](#62-train)
  - [6.3 Private test](#63-private-test)
- [7. Đánh giá kết quả](#7-đánh-giá-kết-quả)
  - [7.1 Public test](#71-public-test)
  - [7.2 Train](#72-train)
  - [7.3 Private test](#73-private-test)
  - [7.4 Đánh giá file khác](#74-đánh-giá-file-khác)
  - [7.5 Đánh giá riêng WER của ASR](#75-đánh-giá-riêng-wer-của-asr)
- [8. Chạy JupyterLab](#8-chạy-jupyterlab)
- [9. Đóng gói và chạy Docker](#9-đóng-gói-và-chạy-docker)
- [10. Fine-tuning và model ASR](#10-fine-tuning-và-model-asr)
  - [10.1 Qwen3.5-4B](#101-qwen35-4b)
  - [10.2 Gipformer](#102-gipformer)

## 1. Tổng quan

LifeMate xử lý yêu cầu thoại tiếng Việt, ánh xạ yêu cầu thành thao tác trên các công cụ của môi trường mô phỏng, và lưu kết quả thành JSON Lines (JSONL). Các lệnh dưới đây giả định đang chạy từ thư mục gốc của repository.

### 1.1 Pipeline

```text
Dữ liệu split + WAV
        │
        ├── ASR Gipformer 1.5 (ONNX, CPU, sherpa-onnx)
        │       └── Lời yêu cầu hiện tại và các lượt user trước
        │
        └── Lịch sử hội thoại + context + tool catalog
                    │
                    └── Qwen3.5-4B FP8 qua vLLM (GPU, OpenAI-compatible API)
                                │
                                ├── ToolEnv kiểm tra schema và thực hiện tool call
                                └── JSONL: id, tool_calls, state_hash, time (ms)
```

1. Dataset loader đọc `{split}.jsonl`, tool catalog và `zaic_tools.ToolEnv` từ thư mục split.
2. ASR đọc đường dẫn các file WAV từ mẫu và lịch sử, nhận dạng lời nói thành văn bản rồi thêm ngữ cảnh hội thoại.
3. Agent gửi tin nhắn và tool schemas đến vLLM. Mỗi tool call được kiểm tra bằng `ToolEnv.validate` trước khi thực thi.
4. Pipeline ghi một dòng cho mỗi mẫu. `time` là thời gian xử lý mẫu tính bằng mili giây.

Trong các mẫu train có `audio_text`, pipeline dùng transcript nhãn khi `prediction.eval_ids` để trống. Khi chạy train để kiểm tra khả năng nhận dạng âm thanh, đặt `prediction.eval_ids` để pipeline dùng ASR cho các ID đó. Public test không có transcript nhãn và được nhận dạng từ WAV.

### 1.2 Tech Stack

| Giai đoạn | Tech Stack | Chức năng |
| --- | --- | --- |
| ASR suy luận | Gipformer 1.5 `gipformer1.5-zac-ft06`, 68M tham số, ONNX, `sherpa-onnx` | Modified beam search, chạy CPU. Dùng audio WAV 16 kHz mono. Thêm hotword từ lịch sử hội thoại. |
| LLM suy luận | Qwen3.5-4B FP8 dynamic safetensors (`RedHatAI/Qwen3.5-4B-FP8-dynamic`) | Chạy bằng vLLM, qua Chat Completions API cục bộ và Qwen tool-call parser. |
| Điều phối agent | Python, `src/zalo_ai` | Xây dựng prompt và danh sách công cụ, điều phối tool loop, ghi JSONL. |
| Công cụ mô phỏng | `zaic_tools.ToolEnv` do bộ dữ liệu cung cấp | Kiểm tra tham số, thực thi lệnh, cung cấp state hash. |
| Huấn luyện Qwen | Unsloth, LoRA, Transformers Trainer, Datasets, PyTorch | Whole-trajectory SFT và xuất adapter bằng `scripts/finetune.sh`. |
| Môi trường | Python 3.12 trở lên, Bash, Docker, NVIDIA CUDA | JupyterLab gọi cùng pipeline dự đoán như CLI. |

## 2. Cấu trúc dự án và dữ liệu

```text
.
├── config.toml                          # Giá trị cấu hình mặc định tập trung
├── scripts/
│   ├── load_config.sh                   # Nạp cấu hình chung cho launcher shell
│   └── finetune.sh                      # Chạy SFT theo config.toml
├── src/
│   ├── lifemate_config.py               # Đọc config.toml và biến môi trường
│   ├── asr/
│   │   ├── api.py                      # API ASR, nạp và dùng lại model
│   │   ├── backends.py                 # Backend sherpa-onnx và Transformers
│   │   ├── config.py                   # Model ASR mặc định và đường dẫn
│   │   ├── validation_ids.txt          # 200 ID train dùng cho validation ASR
│   │   └── evaluate.py                 # WER qua API ASR thực tế
│   └── zalo_ai/
│       ├── dataset.py                  # Đọc split và khởi tạo ToolEnv
│       ├── agent.py                    # Prompt, tool schemas, tool loop
│       ├── pipeline.py                 # Điều phối ASR/LLM, ghi kết quả
│       └── finetune.py                 # Whole-trajectory agent SFT
├── saved_models/
│   ├── asr/gipformer1.5-zac-ft06/      # Trọng số ASR ONNX
│   └── llm/Qwen3.5-4B/                 # Trọng số Qwen FP8 cho vLLM
├── predict.sh                          # Khởi động vLLM rồi chạy dự đoán
├── predict.py                          # Entry point dự đoán Python
├── predict_notebook.ipynb              # Chạy pipeline từ Jupyter
├── start_vllm.sh                       # Cấu hình và khởi động vLLM
├── start_jupyter.sh                    # Khởi động vLLM rồi JupyterLab
├── evaluate.py                          # Đánh giá file dự đoán trên split đã chọn
├── finetune_qwen35_4b_v1.ipynb         # Notebook thử nghiệm cũ
├── requirements-infer.txt              # Dependencies suy luận
├── requirements-finetune.txt           # Dependencies fine-tune Qwen
└── Dockerfile / .dockerignore          # Image suy luận offline
```

Trọng số LLM, ASR và các thư mục adapter hoặc checkpoint được tạo khi fine-tune là model artifacts, không phải mã nguồn. Chạy suy luận cần các thư mục Qwen và Gipformer ở `saved_models/` như cấu hình mặc định hoặc điều chỉnh biến model tương ứng. `paths.llm_model` trỏ tới checkpoint Qwen gốc. `paths.llm_lora_adapter` trỏ tới adapter tùy chọn, để trống nếu muốn chạy model gốc.

Dataset được mount riêng, mặc định ở `/data`:

```text
/data/
├── train/
│   ├── train.jsonl
│   ├── train_tools_catalog.json
│   ├── zaic_tools/
│   └── audio/...
├── public_test/
│   ├── public_test.jsonl
│   ├── public_test_tools_catalog.json
│   ├── zaic_tools/
│   └── audio/...
└── private_test/                 # Được cung cấp vào final test
```

Mỗi split phải có `{split}.jsonl`, `{split}_tools_catalog.json` và package `zaic_tools` cần thiết để mở `ToolEnv`. Cấu hình `prediction.split` chọn đúng một thư mục theo dạng `{paths.data_root}/{prediction.split}`.

## 3. Chuẩn bị môi trường

Các giá trị mặc định nằm trong [`config.toml`](config.toml). Dùng lệnh `python` trong môi trường đã kích hoạt cho các lệnh Python.

### 3.1 Yêu cầu

- Python >= 3.12, CUDA 12.9.
- GPU NVIDIA tối thiểu 12GB VRAM chạy Qwen3.5-4B FP8 cùng với vLLM (ASR mặc định chạy CPU).
- Mô hình + Trọng số của Qwen và Gipformer nằm tại `saved_models/`.
- Bộ dữ liệu mặc định tại `/data`. Đổi đường dẫn bằng `paths.data_root` trong TOML.

### 3.2 Cài dependencies inference

Các thư viện được cài đặt tương thích với CUDA 12.9

```bash
python -m venv .venv-infer
source .venv-infer/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-infer.txt
```

## 4. Cấu hình

Trước khi chạy, kiểm tra các đường dẫn dữ liệu và model trong [`config.toml`](config.toml). Mỗi split cần có JSONL, tool catalog và thư mục `zaic_tools`. Đặt `paths.data_root` nếu dữ liệu không nằm ở `/data`, cập nhật đường dẫn model nếu trọng số không nằm trong `saved_models/`, và chọn `paths.result_dir` nếu `/result` không dùng được. Các launcher nạp cấu hình này tự động. Mô tả đầy đủ từng khóa cùng biến môi trường tương ứng nằm trong [src/lifemate_config.py](src/lifemate_config.py).

## 5. Khởi động vLLM

`start_vllm.sh` khởi động server bằng các thiết lập trong `config.toml`. `predict.sh` tự gọi launcher này, nên chỉ cần khởi động thủ công khi muốn kiểm tra server riêng.

```bash
bash start_vllm.sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
```

Log nằm ở `/tmp/vllm.log`. Mô tả các thiết lập model và server nằm trong [src/lifemate_config.py](src/lifemate_config.py).

## 6. Chạy dự đoán

Trước khi chạy, kiểm tra các yêu cầu trong mục 4, chọn `prediction.split` và tên output trong `config.toml`. `predict.sh` khởi động vLLM rồi chạy dự đoán. Tên output phải là `submission.jsonl` hoặc bắt đầu bằng `submission_` và kết thúc bằng `.jsonl`. Script dừng nếu file đích đã tồn tại.

### 6.1 Public test

Đảm bảo dữ liệu nằm trong thư mục `public_test` bên dưới đường dẫn `paths.data_root`. Với giá trị mặc định, thư mục đó là `/data/public_test`.

```toml
[prediction]
split = "public_test"
eval_ids = ""
submission_file_name = "submission_public.jsonl"
```

```bash
bash predict.sh
```

Kết quả mặc định được ghi vào `/result/submission_public.jsonl`.

### 6.2 Train

Để chạy toàn bộ split train, đặt cấu hình sau. Khi `eval_ids` rỗng, pipeline dùng transcript `audio_text` có trong dữ liệu train.

```toml
[prediction]
split = "train"
eval_ids = ""
submission_file_name = "submission_train.jsonl"
```

```bash
bash predict.sh
```

Để chạy ASR trên các ID train cụ thể, đặt `eval_ids` thành file text có một ID trên mỗi dòng. File validation có sẵn trong repository, hoặc tạo file ID riêng.

```toml
[prediction]
split = "train"
eval_ids = "src/asr/validation_ids.txt"
submission_file_name = "submission_train_asr.jsonl"
```

Chạy lại `bash predict.sh`. Khi có `eval_ids`, pipeline dùng ASR cho các ID đã chọn.

### 6.3 Private test

Trước khi chạy, mount dữ liệu vào thư mục `private_test` bên dưới đường dẫn `paths.data_root`. Thư mục cần có `private_test.jsonl`, `private_test_tools_catalog.json` và package `zaic_tools`.

```toml
[prediction]
split = "private_test"
eval_ids = ""
submission_file_name = "submission_private.jsonl"
```

```bash
bash predict.sh
```

## 7. Đánh giá kết quả

Trước khi đánh giá, đặt `prediction.split`, `prediction.eval_ids` và `prediction.submission_file_name` khớp với lần dự đoán. `evaluate.py` mặc định đọc file trong `paths.result_dir` theo tên đã chọn.

### 7.1 Public test

Đặt split là `public_test`, `eval_ids = ""`, và tên file vừa tạo, ví dụ `submission_public.jsonl`, rồi chạy:

```bash
python evaluate.py
```

Evaluator báo coverage, ID thiếu hoặc trùng, ID ngoài split và thời gian. Test không có nhãn nên không có điểm chính xác tool call.

### 7.2 Train

Đặt split là `train` và dùng đúng `eval_ids` của lần dự đoán. Với dự đoán toàn bộ train, đặt `eval_ids = ""` và tên file `submission_train.jsonl`. Với validation ASR, đặt `eval_ids = "src/asr/validation_ids.txt"` và tên file `submission_train_asr.jsonl`.

```bash
python evaluate.py
```

Evaluator phát lại tool calls và báo điểm, số mẫu, thời gian trung bình cùng số mẫu vượt 30.000 ms.

### 7.3 Private test

Đặt split là `private_test`, `eval_ids = ""`, và tên file vừa tạo, ví dụ `submission_private.jsonl`, rồi chạy:

```bash
python evaluate.py
```

Evaluator báo coverage, ID thiếu hoặc trùng, ID ngoài split và thời gian. Test không có nhãn nên không có điểm chính xác tool call.

### 7.4 Đánh giá file khác

Có thể truyền đường dẫn file dự đoán làm đối số. Split vẫn lấy từ `prediction.split` trong `config.toml`.

```bash
python evaluate.py /result/jupyter_submission.jsonl
```

### 7.5 Đánh giá riêng WER của ASR

`python -m src.asr.evaluate` chạy 200 hội thoại validation cố định qua API ASR và lưu CSV cùng JSON vào `results/asr/`. Script cần thêm `pandas` ngoài các package trong requirements ASR:

```bash
python -m pip install -r src/asr/requirements-sherpa.txt
python -m pip install pandas
python -m src.asr.evaluate --model gipformer_zac
```

Lệnh này đo WER của ASR. Để chấm tool calls đầu cuối, dùng `evaluate.py`.

## 8. Chạy JupyterLab

`start_jupyter.sh` đọc cấu hình, khởi động vLLM nếu cần, rồi chạy JupyterLab. Có thể sửa cấu hình Jupyter trong `config.toml` trước khi khởi động.

```bash
bash start_jupyter.sh
```

Mở địa chỉ `http://localhost:9777` theo mặc định và dùng token `zac2026`. Mở `predict_notebook.ipynb` rồi chạy các cell theo thứ tự sau.

1. Cố định seed và đọc `config.toml`.
2. Nạp ASR Gipformer. Launcher đã khởi động Qwen qua vLLM.
3. Đọc split đã chọn, kiểm tra danh sách ID và xem số mẫu sẽ chạy.
4. Chạy dự đoán bằng cùng các hàm mà CLI sử dụng, in tiến độ và ghi file theo `jupyter.output_file_name` trong `paths.result_dir`. Chạy lại bước này sẽ ghi đè file kết quả.

Để giới hạn notebook vào một nhóm mẫu train, đặt `prediction.split = "train"` và `prediction.eval_ids` thành đường dẫn file ID như ở mục 6.2. Sau khi sửa cấu hình, khởi động lại launcher và kernel để chúng nhận giá trị mới.

## 9. Đóng gói và chạy Docker

Dockerfile dùng base image `vllm/vllm-openai:v0.32.0-cu129`, cài các dependencies inference, sao chép mã và trọng số vào image, đặt chế độ offline, rồi chạy `predict.sh` theo mặc định.

Build image tại thư mục gốc repository:

```bash
docker build -t lifemate:latest .
```

Mount dữ liệu và output, cấp quyền truy cập GPU:

```bash
docker run --rm --gpus all \
  -v /data:/data \
  -v /result:/result \
  lifemate:latest
```

Image dùng bản `config.toml` được sao chép lúc build. Với cấu hình mặc định, trọng số ASR và Qwen phải hiện diện trong `saved_models/`, và kết quả được ghi vào `/result/submission.jsonl`. Nếu đổi `paths.data_root` hoặc `paths.result_dir`, điều chỉnh đường dẫn mount bên trong container cho khớp.

## 10. Fine-tuning và model ASR

### 10.1 Qwen3.5-4B

Pipeline SFT chính nằm ở `src/zalo_ai/finetune.py`. Mỗi mẫu train là một trajectory gồm yêu cầu, các lượt assistant gọi tool và kết quả tool. Chỉ các token assistant thuộc request có nhãn được tính loss. Pipeline kiểm tra prefix với chat template dùng khi inference, chia train/validation và lưu adapter tốt nhất theo validation loss. Các mẫu không có tool call trong nhãn cũng không có câu trả lời assistant, nên được bỏ qua thay vì tự tạo nhãn.

Tạo và kích hoạt môi trường CUDA riêng, sau đó cài dependencies:

```bash
python -m venv .venv-qwen-finetune
source .venv-qwen-finetune/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-finetune.txt
```

Chỉnh `[paths]`, `[prediction].agent_thinking_mode` và `[finetune]` trong `config.toml` trước khi chạy. `paths.data_root/train` phải có `train.jsonl` và `train_tools_catalog.json`; `paths.llm_model` trỏ tới base model. `finetune.output_dir` là thư mục lưu adapter; `sample_limit = 0` dùng toàn bộ dữ liệu, số dương dùng số mẫu đầu để thử. `resume_from_checkpoint = ""` bắt đầu lượt train mới; đặt đường dẫn checkpoint khi cần tiếp tục. Tất cả siêu tham số LoRA, batching, optimizer, lịch học, epoch, early stopping và trọng số fuzzy đều đọc từ `[finetune]`. Thinking của train dùng chung `prediction.agent_thinking_mode` để giữ định dạng đúng lúc inference.

Chạy từ thư mục gốc repository:

```bash
bash scripts/finetune.sh
```

Script dùng `python` của môi trường đang kích hoạt và không nhận tham số dòng lệnh. Mặc định train tối đa 10 epoch, early stopping patience 5, lưu adapter vào `saved_models/qwen35_4b_zaloai_lora/`. Với GPU 24 GiB và trajectory dài, đặt cả hai giá trị `per_device_train_batch_size` và `per_device_eval_batch_size` bằng 1; tăng `gradient_accumulation_steps` để giữ effective batch. `loss_chunk_size` giới hạn số token assistant được tính cross entropy trong mỗi lượt chiếu float32. Pipeline chỉ tạo logits tại vị trí assistant có nhãn, nhưng độ dài trajectory vẫn quyết định bộ nhớ activation. Sau khi train, đặt `paths.llm_lora_adapter` trùng với `finetune.output_dir` để `start_vllm.sh` nạp adapter; dừng vLLM đang chạy trước khi khởi động lại. Đặt `paths.llm_lora_adapter = ""` để infer với base model.

Checkpoint Qwen cục bộ hiện dùng compressed FP8. Vì vậy `finetune.quantization = "bf16"` dùng LoRA 16-bit theo cách Unsloth nạp checkpoint này. Muốn dùng NF4 QLoRA, cần chọn base model chưa lượng tử hóa qua `paths.llm_model`, rồi đặt `finetune.quantization = "4bit"`. Pipeline từ chối cấu hình 4-bit trên checkpoint đã lượng tử hóa để tránh âm thầm chạy sai chế độ.

### 10.2 Gipformer

Model ASR dùng trong pipeline là Gipformer 1.5 `gipformer1.5-zac-ft06`, chạy qua `sherpa-onnx` trên CPU. Trọng số ONNX có trong `saved_models/asr/gipformer1.5-zac-ft06/`.

Để đánh giá WER của model qua đúng API ASR mà pipeline sử dụng, cài requirements của sherpa-onnx, cài thêm `pandas`, rồi chạy:

```bash
python -m pip install -r src/asr/requirements-sherpa.txt
python -m pip install pandas
python -m src.asr.evaluate --model gipformer_zac
```

Checkpoint Gipformer được fine-tune từ `g-group-ai-lab/gipformer1.5-68M-rnnt` và xuất sang ONNX. Mã huấn luyện nằm ngoài repository này. Đặt `paths.gipformer_model` trong `config.toml` để chọn thư mục model khác khi chạy pipeline. Hướng dẫn ASR chi tiết nằm trong [`src/asr/QUICKSTART_GIPFORMER.md`](src/asr/QUICKSTART_GIPFORMER.md).
