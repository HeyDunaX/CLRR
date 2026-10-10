> **Công cụ migration cũ đã hoàn thành và được dọn (10/10/2026).** Dùng [workflow chung](EXPERIMENT_WORKFLOW.md) để upload/verify run mới. Mappings và receipts migration vẫn giữ trong outputs; các tên script migration dưới đây chỉ là tham chiếu lịch sử.

# Lưu checkpoint trên Hugging Face

## Kho chính thức

**Repo public duy nhất cho checkpoint của dự án:** [FiveC/amis-rewire-checkpoints](https://huggingface.co/FiveC/amis-rewire-checkpoints/tree/main).

Không tạo repo private riêng theo đợt chạy. Tên đường dẫn mới không chứa ngày tháng. Thời điểm chạy được ghi trong metadata, không dùng làm tên cấu hình.

## Cấu trúc

Quy tắc hiện hành: mở phương pháp là thấy seed. Chỉ khi cùng seed có nhiều cấu hình mới thêm tầng cấu hình bên trong seed. Các mục migration phía dưới mô tả lịch sử; đường dẫn hiện hành tra ở `registry/checkpoint_index.json` và `registry/seed_layout_mapping.json`.

```text
checkpoints/
  <dataset>/
    <backbone>/
      <method>/
        seed-<seed>/
          [<configuration>/]  # chỉ khi cùng seed có nhiều cấu hình
            best.zip
            metrics.json
            run_metadata.json  # bắt buộc với run mới
            validation_predictions.csv
            test_metrics.json
            test_predictions.csv
            provenance/
            selected_provenance/
            selection/checkpoint-<step>/
registry/
  path_mapping.json
  migration_receipts.json
  imports/
```

| Thành phần | Quy tắc |
|---|---|
| Dataset | Trùng tên `data_processed/<dataset>`: `amis_mandarin`, `opus100_turkish_english_20k` |
| Backbone | `mbart-large-50`; dùng tên khác khi thực sự đổi backbone |
| Method | `baseline`, `middle-alignment`, `lsr`, `clrr-encoder`, `full`; các nhánh khác ghi rõ `clrr-decoder`, `full-decoder`, `full-both`, `bitfit`, `lora`, `strong-lora`, `layerskip` |
| Configuration | Chỉ thêm bên trong seed khi có nhiều bản: mô tả khác biệt phương pháp hoặc protocol; không có ngày tháng |
| Recipe / cohort | Ghi trong metadata: tham số, protocol và mục đích run; không đặt trước seed |
| Seed | `seed-42`, `seed-45`, ... |
| Weights đã chọn | Luôn `best.zip`; selected epoch/step và SHA-256 nằm trong receipt |

Ví dụ Full Turkish:

```text
checkpoints/opus100_turkish_english_20k/mbart-large-50/full/
seed-42/best.zip
```

`0p10` biểu diễn `0.10`; `bs32` là microbatch 32, `ga4` là tích lũy 4, không phải đổi batch. `es10` là patience 10. Recipe Amis đã chạy dùng `es4`; không đổi tên thành `es10`. Độ dài, warmup, weight decay, language codes, dataset/model revision và công thức loss phải nằm trong `training_receipt.json`.

### Metadata của run mới

Mỗi run mới có `run_metadata.json`: dataset và split/revision/hash, backbone/revision, method và tham số, seed/data seed, LR, microbatch/accumulation/effective batch, precision, max epochs, patience, best epoch/step, tiêu chí chọn checkpoint, code revision, protocol đánh giá và SHA-256 archive. Giữ receipts gốc của run cũ; không điền giá trị thiếu bằng suy đoán. `provenance/layout_receipt.json` chỉ chứng minh di chuyển và nguồn seed, không thay thế metadata huấn luyện.

Nếu cùng seed có các bản khác cấu hình/implementation/protocol, lưu dưới tên cấu hình bên trong seed, chẳng hạn `full/seed-42/full_a010_l010-accumfixed/`. Strong LoRA A/B dùng hai nhãn phương pháp rõ ràng `strong-lora-a`, `strong-lora-b`; các đối sánh LSR λ khác được ghi rõ `lsr-lambda0p03`, `lsr-lambda0p30` trong index. Không chọn bản mặc định bằng test score và không ghi đè archive khác hash. Recipe, cohort và nguồn cũ luôn còn trong registry/receipt.

## Quy trình upload và dọn

1. Chọn checkpoint theo validation đã khóa; test không tham gia chọn checkpoint.
2. Upload thẳng từ máy GPU vào đúng thư mục public. Upload weights trước, rồi metadata và predictions.
3. Kiểm tra kích thước và SHA-256 của LFS/Xet ở revision đích đã pin. File Git nhỏ kiểm tra blob SHA-1 Git với bytes nguồn. Kiểm tra digest weights được chọn và weights export khớp nhau.
4. Ghi collection receipt và ánh xạ nếu chuyển từ đường dẫn cũ. Không ghi đè file khác nội dung; cấu hình hoặc cohort khác phải dùng đường dẫn khác.
5. Chỉ xóa weights tạm trên GPU/máy và repo nguồn khi toàn bộ file cần giữ đã được xác minh tại đích. Lỗi upload phải giữ runtime/weights để khôi phục.
6. Thu JSON/CSV/Markdown nhỏ về repo local; tắt GPU sau khi hoàn tất công việc và xác minh backup.

**Không cần tải checkpoint hàng loạt về máy để chuyển repo.** Dùng copy giữa repo trên Hub; đối chiếu LFS SHA-256, kích thước và Git blob ID trước khi xóa nguồn.

## Di chuyển các repo riêng đã tạo

Scripts (chạy trong conda `clrr`):

Đây là lệnh của đợt migration đã hoàn tất. Inventory và mapping của đợt này là snapshot để truy vết, không ghi đè bằng một inventory trống sau khi đã xóa source. Các launcher lịch sử còn tên repo riêng phải được cập nhật sang quy tắc public này trước khi dùng cho đợt training mới.

```powershell
conda run --no-capture-output -n clrr python -X utf8 scripts/revalidation/inventory_hf_checkpoints.py
conda run --no-capture-output -n clrr python -X utf8 scripts/revalidation/consolidate_hf_checkpoints.py
conda run --no-capture-output -n clrr python -X utf8 scripts/revalidation/consolidate_hf_checkpoints.py --apply --delete-sources
```

Inventory đọc source revision; mapping kiểm tra xung đột trước khi copy. Lần chạy có `--apply` thực hiện quyền public mà người dùng đã cấp. `--delete-sources` chỉ xóa sau khi xác minh các file đích và bảo đảm revision nguồn không thay đổi.

SDK hỗ trợ copy giữa repo được cài riêng trong `scratch/hf_migration_sdk` để không thay dependency của môi trường train. Không in token; dùng token đã được cấp quyền trong launcher.

Artifacts nhỏ: `outputs_rebuttal/hf_checkpoint_consolidation/{inventory,path_mapping,migration_receipts}.json`. Registry trên HF giữ mapping và receipts. Khi đọc tài liệu lịch sử, dùng mapping để tìm file ở repo mới; nội dung số liệu và receipt gốc được giữ để truy vết. Tên cũ có ngày tháng bên trong metadata là thông tin nguồn, không phải quy tắc đặt tên mới.

## Đợt gom hiện tại

11 repo private của CLRR được gom: Amis accumulation confirmation, hai corrected controls, bốn fresh confirmation families, bốn Turkish pilot families. Không dùng tên thư mục để suy luận rằng một run đã hoàn tất; kiểm tra `VAL_COMPLETE.json`/`COMPLETE.json` và collection receipt.

**Trạng thái: HOÀN TẤT.** Đã chuyển và xác minh **11 repo / 2.242 file**, bao gồm **42 archive checkpoint cũ**. Sau xác minh, đã xóa 11 repo nguồn riêng. Hai archive Baseline/Middle Turkish mới được upload trực tiếp từ Colab vào cấu trúc public và xác minh riêng; tổng cộng **44 archive** thuộc phần được gom trong đợt này. Các file tải tạm về máy đã được xóa sau xác minh bản public.

Nội dung số liệu và metadata gốc không đổi trong bước di chuyển; registry giữ source repo/revision/path và path mới để giải quyết các tham chiếu lịch sử. Test Turkish đã hoàn tất cho cả bốn family; bảng số ở [TURKISH_MBART_SEED42_RESULTS.md](../experiments/EXPERIMENTS_UPDATED.md#turkish-seed42). Hai A100 đã tắt, mức tiêu thụ xác minh cuối là 0 unit/giờ.

### Upload/test trực tiếp từ Colab cho Turkish

`scripts/revalidation/upload_turkish_controls_public.py` cứu upload của hai control đã train xong, kiểm tra weights selected/exported và archive SHA-256, upload archive/metadata/predictions vào repo public. `scripts/revalidation/test_turkish_public_store.py` khóa bốn selections thật từ repo public, xác minh digest/receipt, rồi dùng đúng worker đánh giá đã kiểm tra để chạy test. Các script này không training lại.

`scripts/revalidation/collect_turkish_public_results.py` chỉ tải receipts/metrics/predictions nhỏ; `verify_turkish_public_receipts.py` kiểm tra đủ 2.000 source/target đúng thứ tự và tính lại cả ba metric. File `collection_receipt.json` cuối của từng family chứng minh archive public và test artifacts ở một revision đã pin.

Controller phải xác nhận đúng runtime A100 tồn tại trước SSH. Thử SSH tới alias đã dừng có thể tự tạo runtime mới qua CLI; không dùng thao tác đó để kiểm tra máy đã tắt. Khi worker lỗi upload, giữ weights và runtime để khôi phục; metadata nhỏ đã thu không đủ để cho phép xóa weights chưa backup.

## Chuẩn hóa toàn bộ kho cũ và chỉ giữ checkpoint cuối cùng

**Hoàn tất dọn cây file:** đã chuyển và xác minh 823 file lịch sử cần giữ; bỏ 190 archive thử/trung gian, tương đương 678.418 GB dung lượng file logic ở các đường dẫn bị bỏ. Kho hiện hành có **90 archive cuối cùng**: 44 từ đợt gom repo riêng/đánh giá Turkish, 46 từ kho cũ. Trong 46 bản cũ: 28 mBART, 7 mT5, 5 ByT5, 6 NLLB. Các checkpoint mới của Turkish đã nằm trong cấu trúc này.

Theo yêu cầu mới, weights thử cấu hình, warmup, replica, attached-anchor control và checkpoint trung gian được bỏ khỏi cây file hiện hành. Giữ checkpoint cuối cùng của các run kết quả chính, đối sánh, ablation cần dùng và xác nhận. JSON/CSV/Markdown nhỏ của các bản thử vẫn được giữ để truy vết; sự hiện diện của report không có nghĩa weights của bản thử còn tồn tại.

Các nhóm cũ `checkpoints/<run>`, `comparative_baselines`, `peft_baselines`, `nllb-200`, `revalidation_ablations`, các đợt Amis và strong LoRA đều được xét. File metrics rời có đủ thông tin được đặt cạnh model tương ứng. Các báo cáo chung nằm dưới `registry/reports/`. Thư mục gốc chỉ giữ `.gitattributes`, `README.md`, `checkpoints/`, `registry/`.

### Tra cứu

- [Bảng checkpoint cuối cùng](https://huggingface.co/FiveC/amis-rewire-checkpoints/blob/main/registry/CHECKPOINT_INDEX.md).
- [CSV để lọc dataset/backbone/method/recipe/cohort/seed](https://huggingface.co/FiveC/amis-rewire-checkpoints/blob/main/registry/checkpoint_index.csv).
- `registry/legacy_path_mapping.json`: đường dẫn cũ → mới, revision nguồn và hash.
- `registry/legacy_discarded_weights.json`: từng archive đã bỏ và lý do.
- `registry/legacy_migration_receipt.json` và `registry/LEGACY_FINAL_AUDIT.json`: bằng chứng xác minh.

Các run lịch sử thiếu cấu hình đầy đủ dùng `historical-<tên-run-không-ngày>` ở vị trí recipe; seed chỉ ghi số khi có trong tên run hoặc metadata gốc, nếu không ghi `seed-unknown`. Đây là nhãn thiếu bằng chứng, không phải giá trị seed mặc định. Những đường dẫn này không được dùng để suy đoán protocol hay tính fair giữa các thí nghiệm. Run mới phải ghi đầy đủ recipe và training receipt theo quy tắc ở đầu tài liệu.

### Gói tổng hợp NLLB

`nllb_results.zip` chứa weights trùng nhóm và checkpoint/optimizer trung gian. Trước khi bỏ gói này, trích riêng JSON/CSV/Markdown nhỏ bằng HTTP Range, kiểm tra ZIP CRC, upload và xác minh Git blob/LFS hash. Giữ sáu archive cuối cùng riêng biệt của NLLB; file metrics/predictions đi cùng từng model, trainer state/config từ gói tổng hợp nằm trong `provenance/aggregate/`. Danh sách nội dung nguồn và receipts nằm ở `registry/nllb_aggregate_contents.json` và `registry/nllb_extraction_receipts.json`. Không tải gói 41 GB về máy.

### Script thực hiện

```powershell
conda run --no-capture-output -n clrr python -X utf8 scripts/revalidation/organize_legacy_hf_store.py
conda run --no-capture-output -n clrr python -X utf8 scripts/revalidation/organize_legacy_hf_store.py --apply
conda run --no-capture-output -n clrr python -X utf8 scripts/revalidation/index_public_checkpoint_store.py
```

Đây là script migration cho snapshot hiện tại, không phải công cụ tùy ý xóa các run mới. Script giữ inventory nguồn cố định, kiểm tra xung đột, copy trên Hub, xác minh toàn bộ file cần giữ, sau đó mới xóa đường dẫn cũ và weights thử được cho phép bỏ. Việc dọn cây file không xóa lịch sử Git; dung lượng logic của những đường dẫn bị bỏ không được coi là dung lượng Hub đã thu hồi vĩnh viễn.

### Khi push run mới

1. Chọn đúng dataset/backbone/method/seed; không thêm ngày tháng vào path. Cohort ghi trong metadata.
2. Khóa recipe thực tế và seed; không ghi đè một run đã tồn tại với weights khác.
3. Upload weights được chọn thành `best.zip`, kèm metrics/predictions và receipt. Nếu lưu thêm checkpoint phục vụ chẩn đoán, phải có mục đích rõ ràng; không mặc định upload mọi epoch.
4. Xác minh archive SHA-256 và receipt rồi mới xóa weights tạm; cập nhật index sau upload.
5. Các path cũ trong receipt lịch sử giữ nguyên để bảo toàn nguồn; dùng mapping để giải quyết tham chiếu. Không sửa số khoa học trong bước dọn kho.


## Cập nhật cấu trúc seed — 08/10/2026

**HOÀN TẤT, xác minh độc lập qua API public.** Đã chuyển 3.056 file và giữ nguyên hash của 90 archive; mở dataset/backbone/method là thấy seed. Có 79 nhóm method/seed; các bản cùng nhóm được giữ trong thư mục cấu hình bên trong seed. Cả 16 archive từng ghi `seed-unknown` đã xác minh seed42 bằng `training_args.bin`, không suy đoán từ tên run. Không train, không đổi metrics, không đổi actual patience.

Các mô tả `historical-*`/`seed-unknown` ở phần migration trước là trạng thái lịch sử. Registry hiện hành: `seed_layout_mapping.json`, `seed_layout_seed_evidence.json`, `SEED_LAYOUT_VERIFIED.json`, `SEED_LAYOUT_PUBLIC_VERIFICATION.json` và `checkpoint_index.json/csv`. `metadata_complete=null` trong index mới nghĩa là bước di chuyển không đánh giá lại tính đầy đủ metadata; xác minh seed không đồng nghĩa đủ endpoint/training provenance.

Công cụ chuyển: `scripts/revalidation/flatten_hf_seed_layout.py`; công cụ kiểm tra: `scripts/revalidation/verify_hf_seed_layout.py`, chạy trong conda `clrr`. Script chuyển dành cho snapshot đã khóa, không chạy lại lên cây đã chuyển. Công cụ index lịch sử yêu cầu cấu trúc cũ; không chạy lại để ghi đè index hiện hành.
