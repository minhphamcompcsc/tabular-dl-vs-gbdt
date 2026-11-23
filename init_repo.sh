#!/usr/bin/env bash
# Tạo lịch sử Git ban đầu cho repo, dưới tên và email Git của chính bạn.
#   main          : khung repo, hợp đồng giao diện, stub cho A và B
#   feature/eval  : phần của C (Minh), chia thành nhiều commit nhỏ
# Chạy một lần, trong thư mục vừa giải nén:  bash init_repo.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ -d .git ]; then
  echo "Thư mục này đã là một Git repo, dừng lại để không ghi đè lịch sử." >&2
  exit 1
fi
if [ -z "$(git config user.name || true)" ] || [ -z "$(git config user.email || true)" ]; then
  echo "Chưa cấu hình Git. Chạy trước:" >&2
  echo '  git config --global user.name  "Tên của bạn"' >&2
  echo '  git config --global user.email "email-dang-ky-github@example.com"' >&2
  exit 1
fi

TRAILER=$'\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
commit() { git add -- "${@:2}" && git commit -q -m "$1$TRAILER" && echo "  $(git log -1 --format='%h %s')"; }

git init -q -b main
echo "Nhánh main:"
commit "chore: khởi tạo repo, cấu trúc thư mục và hợp đồng giao diện giữa A, B, C" \
  .gitignore README.md requirements.txt configs results/.gitkeep \
  src/__init__.py src/interfaces.py src/models/__init__.py tests/__init__.py tests/conftest.py
commit "chore: stub cho phần của A (dữ liệu, GBDT, tuning) và B (mạng nơ-ron, TabPFN)" \
  src/data.py src/tune.py src/models/gbdt.py src/models/nn.py src/models/ft_transformer.py src/models/tabpfn_wrapper.py

git checkout -q -b feature/eval
echo "Nhánh feature/eval:"
commit "feat(eval): metrics ROC-AUC, accuracy, RMSE kèm test" \
  src/metrics.py tests/test_metrics.py
commit "feat(eval): thống kê Wilcoxon theo cặp seed, hiệu chỉnh Holm, xếp hạng trung bình kèm test" \
  src/stats.py tests/test_stats.py
commit "feat(eval): baseline Logistic/Linear regression, dữ liệu giả và test hợp đồng cho mọi mô hình" \
  src/models/baseline.py src/fakes.py tests/test_contract.py
commit "feat(eval): runner ghi CSV theo hợp đồng, resume, ghi phần cứng và commit; các script thí nghiệm" \
  src/runner.py src/config.py src/env_info.py src/results_io.py scripts/_bootstrap.py \
  scripts/run_baseline.py scripts/run_full.py scripts/run_ablation_budget.py scripts/run_size_curve.py scripts/run_tabpfn.py
commit "feat(eval): bảng 2, 3, 4 và hình 2, 3 từ CSV kèm test end-to-end" \
  src/plots.py scripts/make_report.py tests/test_pipeline.py
commit "feat(eval): notebook Colab cho các lần chạy chính thức trên A100" \
  notebooks/run_official.ipynb
commit "ci: chạy pytest và smoke test trên Python 3.10 và 3.12" \
  .github/workflows/tests.yml
git add -- init_repo.sh && git commit -q -m "chore: script khởi tạo lịch sử repo$TRAILER"

if [ -n "$(git status --porcelain)" ]; then
  echo "Cảnh báo: còn file chưa commit:" >&2
  git status --short >&2
fi

cat <<'MSG'

Xong. Bước tiếp theo để đưa lên GitHub (chọn một cách):

  # Cách 1: dùng GitHub CLI
  gh repo create tabular-dl-vs-gbdt --private --source . --remote origin
  git push -u origin main feature/eval

  # Cách 2: tạo repo trống trên github.com rồi
  git remote add origin https://github.com/<tai-khoan>/tabular-dl-vs-gbdt.git
  git push -u origin main feature/eval

Sau đó mở pull request feature/eval -> main và nhờ Huấn hoặc Thu review.
MSG
