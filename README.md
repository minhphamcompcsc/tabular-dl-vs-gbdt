# Deep learning vs GBDT trên dữ liệu bảng (CO5175, Topic 3)

Bài tập lớn môn Học máy và Ứng dụng (CO5175). Câu hỏi chính: trên dữ liệu bảng, khi so sánh công bằng (cùng cách chia dữ liệu, cùng ngân sách tuning, nhiều seed, có kiểm định thống kê), mạng nơ-ron có thắng GBDT không?

- **Dataset:** Adult, California Housing, German Credit
- **Mô hình:** XGBoost, LightGBM, CatBoost; MLP, ResNet, FT-Transformer; TabPFN (mở rộng); Logistic/Linear regression (baseline)
- **Paper chính:** Ye et al. (2024), *A Closer Look at Deep Learning Methods on Tabular Datasets*, https://arxiv.org/abs/2407.00956

## Phân công và trạng thái

| Thành viên | Phạm vi | File | Trạng thái |
| --- | --- | --- | --- |
| A (Huấn) | Dữ liệu, GBDT, tuning | `src/data.py`, `src/models/gbdt.py`, `src/tune.py`, `configs/datasets.yaml`, `configs/search_spaces/` | Đã triển khai và kiểm thử |
| B (Thu) | Mạng nơ-ron, TabPFN | `src/models/nn.py`, `src/models/ft_transformer.py`, `src/models/tabpfn_wrapper.py`, `configs/search_spaces/` | Stub, chờ cài đặt |
| C (Minh) | Đánh giá, thống kê, script, tài liệu | `src/metrics.py`, `src/stats.py`, `src/plots.py`, `src/runner.py`, `src/models/baseline.py`, `scripts/`, `notebooks/`, `tests/` | Đã có bản đầu |

"Stub" nghĩa là file đã có đúng tên hàm, tên lớp và tham số theo hợp đồng bên dưới, nhưng thân hàm chỉ báo `NotImplementedError`. Người phụ trách chỉ cần thay thân hàm. Khi một mô hình còn là stub, các script tự bỏ qua mô hình đó và in `[skip]`, không bị dừng.

## Cài đặt

```bash
git clone <REPO_URL> && cd tabular-dl-vs-gbdt
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                              # test hợp đồng, metrics, thống kê, pipeline
```

Yêu cầu Python 3.10 trở lên. Phiên bản chính xác của lần chạy chính thức được ghi vào `results/env/requirements-lock.txt`.

Trên Ubuntu hoặc WSL, cài thư viện OpenMP của hệ thống:

```bash
sudo apt install -y libgomp1
```

## Dữ liệu và tiền xử lý

| Dataset | Nguồn | Số mẫu × đặc trưng | Nhãn |
| --- | --- | --- | --- |
| Adult | [UCI, ID 2](https://archive.ics.uci.edu/dataset/2/adult) | 48.842 × 14 | Thu nhập > 50.000 USD là lớp 1 |
| California Housing | [StatLib qua scikit-learn](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) | 20.640 × 8 | Giá nhà trung vị, đơn vị 100.000 USD |
| German Credit | [UCI, ID 144](https://archive.ics.uci.edu/dataset/144/statlog-german-credit-data) | 1.000 × 20 | Rủi ro tín dụng xấu là lớp 1 |

Dữ liệu được tải tự động bằng `ucimlrepo` và scikit-learn. Split train/validation/test là 60/20/20, phân tầng cho phân loại nhị phân. `train_size` lấy mẫu con từ train sau khi chia. Bộ tiền xử lý học cách điền giá trị thiếu và mã hóa one-hot từ train. Nhánh `nn` chuẩn hóa thêm các cột số.

## Chạy thử GBDT và Optuna trên Ubuntu

Ví dụ này chạy XGBoost trên German Credit với một seed và 3 trial trên CPU. Tạo thư mục tạm để CSV đánh giá nằm ngoài thư mục kết quả chính thức.

```bash
tmp_dir=$(mktemp -d)
python scripts/run_full.py \
  --datasets german_credit \
  --models xgboost \
  --seeds 0 \
  --n-trials 3 \
  --strict \
  --out "$tmp_dir/check.csv"
```

Optuna đọc `configs/search_spaces/<model>.yaml` và chọn tham số theo ROC-AUC validation cho phân loại hoặc RMSE validation cho hồi quy. Runner lưu JSON tham số và CSV lịch sử trial trong `results/best_params/`, rồi đánh giá trên test. Giữ cache 3 trial và CSV của lượt chạy thử ngoài commit kết quả chính thức.

## Chạy thử với dữ liệu giả (không cần phần của A và B)

```bash
python scripts/run_baseline.py --smoke
python scripts/run_full.py --smoke
python scripts/run_ablation_budget.py --smoke
python scripts/run_size_curve.py --smoke
python scripts/run_tabpfn.py --smoke --models dummy,linear   # TabPFN thật chưa cần
python scripts/make_report.py --smoke      # bảng và hình trong results/smoke/
python scripts/make_dataset_table.py --smoke   # Bảng 1 trên dữ liệu giả
```

`--smoke` dùng 2 dataset tổng hợp (`fake_binary`, `fake_regression`), 3 seed, và 2 mô hình thay thế từ scikit-learn (`sk_hgb` thay cho GBDT, `sk_mlp` thay cho mạng nơ-ron). Hai mô hình này chỉ để kiểm tra pipeline, **không đưa vào báo cáo**. Thư mục `results/smoke/` không được commit.

## Chạy thí nghiệm thật

Các lần đo chính thức chạy trên Colab A100 bằng `notebooks/run_official.ipynb` (tài khoản của Huấn), để thời gian huấn luyện so sánh được với nhau. Chạy thử độ chính xác trên máy cá nhân thì dùng lệnh như dưới.

| Lệnh | Mục báo cáo | Đầu ra |
| --- | --- | --- |
| `python scripts/run_baseline.py` | 3.2 Baseline | `results/baseline.csv` |
| `python scripts/run_full.py` | 3.3, 3.4, 3.8 | `results/full.csv`, `results/best_params/` |
| `python scripts/run_ablation_budget.py` | 3.5, Hình 2 | `results/ablation_budget.csv` |
| `python scripts/run_size_curve.py` | 3.6, Hình 3 | `results/size_curve.csv` |
| `python scripts/run_tabpfn.py` | 3.7 | `results/tabpfn.csv` |
| `python scripts/make_report.py` | Bảng 2a (baseline), 2, 3, 4, 5 (TabPFN); Hình 2, 3, 4 (TabPFN) | `results/tables/`, `results/figures/` |
| `python scripts/make_dataset_table.py` | Bảng 1 (mô tả dataset, cần tải dữ liệu) | `results/tables/table1_datasets.*` |

Bảng LaTeX dùng `\toprule`, `\midrule`, `\bottomrule`, nên file `.tex` của báo cáo cần `\usepackage{booktabs}`.

Tùy chọn chung cho mọi script `run_*`:

- `--datasets adult,german_credit`, `--models xgboost,mlp`, `--seeds 0,1,2`, `--n-trials 15`, `--out path.csv`
- `--no-resume`: chạy lại cả những dòng đã có trong CSV (mặc định bỏ qua để chạy tiếp khi Colab ngắt kết nối)
- `--strict`: dừng ngay nếu mô hình chưa cài đặt, thay vì bỏ qua
- `run_full.py --no-tune`: dùng tham số mặc định, ghi `n_trials = 0`

Cấu hình nằm trong `configs/experiment.yaml` (seed, tỉ lệ chia, N, dataset) và `configs/experiments/<tên>.yaml` (mô hình, ngân sách, kích thước train của từng thí nghiệm). Không viết cứng các giá trị này trong code.

## Quy trình một lần chạy

```
load_dataset -> make_split(seed, train_size) -> preprocessor.fit(chỉ trên train)
-> model.fit(train, val) -> evaluate trên test -> mỗi thước đo một dòng CSV
```

Tuning chạy một lần trên split của seed 0 với đúng N lần thử cho mỗi cặp dataset–mô hình được tuning. Tham số tốt nhất được lưu ở `results/best_params/` và dùng lại cho cả 10 seed và các thí nghiệm khác.

## Hợp đồng giữa các phần

Định nghĩa đầy đủ ở `src/interfaces.py`. Đổi gì ở đây phải báo cả nhóm.

| Thành phần | Chữ ký | Người viết |
| --- | --- | --- |
| Tải dữ liệu | `load_dataset(name) -> (X: DataFrame, y: ndarray, meta: DatasetMeta)`; nhãn nhị phân là 0/1 | A |
| Chia dữ liệu | `make_split(X, y, seed, train_size=None) -> Split(train, val, test)`; 60/20/20, phân tầng; `train_size` chỉ lấy mẫu con tập train | A |
| Tiền xử lý | `build_preprocessor(kind, meta)` với `kind` là `"nn"` hoặc `"gbdt"`; có `fit(X_train)` và `transform(X)` | A |
| Mô hình | `__init__(task, seed=0, **params)`, `fit(X_tr, y_tr, X_val, y_val)`, `predict(X)` trả P(y=1) hoặc giá trị hồi quy; thuộc tính `preprocess_kind`, `n_params`, `device` | A (GBDT), B (mạng, TabPFN), C (baseline) |
| Tuning | `tune(model_name, dataset, n_trials, seed=0) -> (best_params: dict, history: DataFrame)` | A |
| Đánh giá | `evaluate(model, split, task) -> dict` | C |
| So sánh thống kê | `compare(df, model_a, model_b, dataset, metric)`, `compare_to_reference(df)` | C |

Mô hình mới phải được đăng ký trong `src/models/__init__.py` (tên, lớp, họ). `tests/test_contract.py` tự kiểm tra mọi mô hình đã đăng ký và `src/data.py` ngay khi chúng được cài đặt: dự đoán đúng kích thước, xác suất nằm trong [0, 1], `n_params > 0`, chia dữ liệu đúng tỉ lệ, cùng seed cho cùng kết quả, `train_size` không làm đổi tập test.

## Định dạng file kết quả

Mỗi file `results/*.csv` có đúng các cột sau, mỗi dòng là một thước đo của một lần chạy:

`dataset, model, seed, n_trials, n_train, metric_name, metric_value, train_time_s, n_params, hardware, git_commit`

- `metric_name`: `roc_auc` và `accuracy` (phân loại), `rmse` (hồi quy). Thước đo chính để xếp hạng và kiểm định: ROC-AUC và RMSE.
- `n_params`: tổng số nút cây đối với ba GBDT. Các mô hình khác báo kích thước theo cách triển khai của từng mô hình.
- `hardware`: tên GPU nếu mô hình đặt `device = "cuda"`, ngược lại là tên CPU.
- `git_commit`: commit của code lúc chạy; có hậu tố `-dirty` nếu `src/`, `configs/`, `scripts/` có thay đổi chưa commit. Kết quả chính thức không được có `-dirty`.

## Kiểm định thống kê

- Wilcoxon signed-rank theo cặp seed (hai phía), hiệu chỉnh Holm trong từng dataset.
- Mặc định so mỗi mô hình deep learning với GBDT tốt nhất trên dataset đó, để số cặp ít và Holm bớt khắt khe.
- Với n seed, p-value nhỏ nhất có thể là 2 / 2^n: 3 seed là 0,25; 5 seed là 0,0625; 10 seed là 0,002. Vì vậy cần giữ 10 seed.
- Hạn chế cần nêu trong báo cáo: chọn GBDT tốt nhất dựa trên chính kết quả test hơi thiên vị về phía GBDT; các lần chia dữ liệu có chồng lấn nên p-value có thể lạc quan.

## Cấu trúc thư mục

```
configs/          experiment.yaml (dùng chung), experiments/*.yaml (từng thí nghiệm), datasets.yaml, search_spaces/
src/              interfaces.py (hợp đồng), data.py, tune.py, metrics.py, stats.py, plots.py,
                  runner.py, config.py, env_info.py, results_io.py, fakes.py, models/
scripts/          run_baseline.py, run_full.py, run_ablation_budget.py, run_size_curve.py, run_tabpfn.py, make_report.py, make_dataset_table.py
notebooks/        run_official.ipynb (chạy chính thức trên Colab)
results/          CSV thô (commit), best_params/, tables/, figures/, env/
tests/            test_contract.py, test_metrics.py, test_stats.py, test_pipeline.py
```

## Quy ước Git

1. Nhánh `main` chỉ nhận code qua pull request, có ít nhất một người khác review.
2. Mỗi người làm trên nhánh riêng: `feature/data` (A), `feature/nn` (B), `feature/eval` (C).
3. Commit nhỏ, thường xuyên, thông điệp rõ ràng; giảng viên có thể xem lịch sử commit.
4. Không commit dataset; dataset được tải bằng code.
5. Gắn tag `v1.0` khi nộp bài.

## Nguồn mã bên ngoài

Chưa dùng mã từ repo công khai. Khi dùng, ghi nguồn ở đầu file và thêm vào bảng này, nêu rõ phần nhóm tự viết hoặc sửa đổi (mục 4.3 của đề).

| File | Nguồn | Phần nhóm sửa đổi |
| --- | --- | --- |
| | | |

## Khai báo công cụ AI

Theo mục 4.4 của đề, mọi công cụ AI phải được khai báo trong phụ lục B của báo cáo.

| Công cụ | Dùng cho phần nào | Người dùng |
| --- | --- | --- |
| Claude (Anthropic) | Dựng khung repo, hợp đồng giao diện, code phần C (metrics, stats, runner, plots, scripts, tests, notebook) | Minh |
| ChatGPT (OpenAI) | Giải thích giao diện, gợi ý mã cho tải dữ liệu, chia tập, tiền xử lý, GBDT và Optuna, hỗ trợ kiểm thử | Huấn |
