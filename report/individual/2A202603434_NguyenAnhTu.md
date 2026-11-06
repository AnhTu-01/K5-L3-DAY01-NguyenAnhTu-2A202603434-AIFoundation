# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:2A202603434
- **Họ và tên: Nguyễn Anh Tú
- **SEED** 4252

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | có |

## 2. Kết quả chính

*Cách tiếp cận	F1 test	F1 SUB 40	p95 (ms/tin)	USD/1k tin
1. BoW + LogReg (tự code)	[ĐIỀN]	0,857	≈ 0,1	0
2. BoW + MLP	[ĐIỀN]	0,889	≈ 0,25	0
3a. Encoder (MiniLM) + kNN	[ĐIỀN]	0,947	≈ 140	0
3b. Encoder (MiniLM) + LogReg	[ĐIỀN]	0,974	≈ 43	0
3c. Encoder (MiniLM) + MLP head	[ĐIỀN]	0,919	≈ 38	0
4a. LLM nhỏ zero-shot (Ollama, qwen2.5:0.5b)	[ĐIỀN]	0,174	≈ 2.900	0 (chạy cục bộ)
4b. LLM nhỏ few-shot	[ĐIỀN]	≈ 0	≈ 7.000	0 (chạy cục bộ)
4c. LLM lớn zero-shot (Ollama, llama3.2:3b)	[ĐIỀN]	≈ 0,86	≈ 9.500	0 (chạy cục bộ).*

## 3. Dự đoán và thực tế

*a. hard_eval (50 case khó).

Mình đoán: [CHÉP LẠI DỰ ĐOÁN]. Ý chính: Encoder sẽ bền hơn BoW, và prompt phòng thủ (4d) chống prompt injection tốt hơn 4a.
Thực tế: mọi model đều tụt mạnh so với test thường (BoW + LogReg 0,857 → 0,378; Encoder + LogReg 0,974 → 0,583). Trong các model Encoder, kNN (3a) lại giữ F1 tốt nhất (0,759), cao hơn cả LogReg (0,583). Hai hàng LLM nhỏ (4a, 4d) gần như bằng 0 vì model luôn trả lời "ham", nên không so sánh được hiệu quả của prompt phòng thủ.
Vì sao lệch: mình đánh giá Encoder + LogReg tốt nhất vì nó cao nhất trên test thường. Nhưng đầu LogReg học ranh giới tuyến tính trên các tin SMS "bình thường", còn kNN dựa vào các tin giống nhất nên bền hơn với dạng tin lạ. Mình cũng không đoán được model 0,5B quá yếu để làm phân loại.

b. llm_fewshot (thêm ví dụ vào prompt).

Mình đoán: [CHÉP LẠI DỰ ĐOÁN]. Ý chính: few-shot sẽ tốt hơn zero-shot.
Thực tế: với model nhỏ trên Ollama, few-shot cho F1 ≈ 0, tệ hơn zero-shot (0,174).
Vì sao lệch: model 0,5B không bám theo định dạng ví dụ, prompt dài hơn làm nó đoán một nhãn cố định, và có thể một phần đầu ra bị cắt do giới hạn token. Few-shot chỉ có ích khi model đủ lớn để học từ ví dụ.

c. emb_tricky / thử thách (Encoder so với BoW trên tin do mình viết).

Mình đoán: Encoder sẽ bắt được spam né từ khoá mà BoW bỏ sót, còn BoW sẽ nhầm ở tin ham có "free"/"win".
Thực tế: cả hai đúng 2/4. Cả hai bỏ sót 2 tin spam né từ khoá (BoW p = 0,21 và 0,09; Encoder p = 0,42 và 0,15) và cùng đúng 2 tin ham có "free"/"win".
Vì sao lệch: mình đánh giá quá cao từ khoá làm BoW sai. Với tin ham có nhiều từ đời thường, một từ như "free" không đủ để kéo xác suất qua 0,5. Điểm yếu thật là spam viết như tin bình thường.*

## 4. Phân tích 50 case khó

*Model tụt nhiều nhất: BoW + LogReg (F1 từ 0,857 xuống 0,378, độ chính xác 54%) và BoW + MLP (0,889 xuống 0,439). Hai hàng LLM nhỏ thấp nhất tuyệt đối (F1 0,067 và 0,000) do luôn đoán "ham".

Nhóm case khó nhất: spam giọng tự nhiên. BoW + MLP, LLM nhỏ đạt 0%, các model Encoder chỉ đạt 10% đến 50%. Nhóm "né từ khoá" cũng làm BoW gục (17% và 0%), nhưng các model Encoder đạt 83%. Nhóm prompt injection thì LLM nhỏ chỉ đạt 17%, trong khi Encoder + LogReg và Encoder + MLP đạt 100%.

Hai case cụ thể:

Case #12 (spam giọng tự nhiên): "hi there, sorry to bother you, my friend gave me your number, I'm new i…" Mở đầu lịch sự, không có từ khoá, dụ vào trang hẹn hò. Cả 7 model đều sai vì dấu hiệu spam nằm ở ý định (dụ click) mà không có từ nào báo hiệu. BoW không thấy từ spam, Encoder thấy giọng thân thiện giống tin ham.
Case #29 (né từ khoá): "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…" Ký tự bị thay bằng số, nên BoW không khớp các từ "congratulations", "won", "gift". Model Encoder đạt 83% ở nhóm này vì tokenizer chia từ con vẫn giữ lại phần nào ý nghĩa. Một bước tiền xử lý chuẩn hoá (đổi 0→o, 4→a, 1→i, bỏ khoảng trắng và dấu chấm xen giữa chữ) sẽ giúp BoW đỡ bị qua mặt.

Về LLM: LLM nhỏ (qwen2.5:0.5b) trả lời "ham" cho gần như mọi tin (4d sai 30/50, tất cả là tin spam thật; độ chính xác 40% bằng đúng tỉ lệ tin ham trong bộ). Vì vậy so sánh 4a với 4d (prompt phòng thủ) trong lần chạy này chưa có giá trị kết luận. Ý tưởng phòng thủ ngoài prompt: không để LLM quyết định một mình, mà kết hợp với bộ phân loại Encoder + LogReg (đạt 100% ở nhóm prompt injection), và chỉ tin LLM khi hai bên đồng ý.

Bộ 50 tin có đại diện cho thực tế không: không. Bộ này chọn cố ý là tin khó, mỗi nhóm khoảng 6 đến 10 tin, nên tỉ lệ spam không phản ánh thực tế (khoảng 13%). Chỉ cần 1 tin đổi kết quả là F1 nhảy vài điểm, nên chênh lệch nhỏ giữa các model chưa chắc có ý nghĩa. Tập chỉ có tiếng Anh theo kiểu SMS Anh.*

## 5. Thử thách cá nhân

*Đề #5 (SEED 4252 mod 6 = 4): viết 4 tin nhắn của riêng mình (2 spam né từ khoá, 2 ham chứa "free"/"win"), so dự đoán BoW với Encoder + LogReg và giải thích từng tin.

Cách làm: mình viết 4 tin tiếng Anh, tính xác suất spam của BoW (vectorize → predict_proba với w, b của Phần 1) và của Encoder + LogReg (emb_model.encode, chuẩn hoá bằng mu, sd, rồi predict_proba với w_e, b_e của 3b). Ngưỡng quyết định 0,5.
Kết quả: BoW đúng 2/4, Encoder + LogReg đúng 2/4.

Giải thích:

Tin 1 và 2 (spam): không có từ spam kinh điển (free, prize, win, claim), và giọng văn như tin thường (chào hỏi, thông báo giao hàng). BoW chỉ cộng trọng số từng từ nên coi là ham. Encoder đọc cả câu và cho p = 0,42 ở tin 1, gần ngưỡng hơn BoW (0,21) nhưng chưa vượt 0,5. Tin 2 nghe giống thông báo giao hàng hợp lệ nên cả hai đều không nghi ngờ.
Tin 3 và 4 (ham có "free"/"win"): cả hai đúng. Một từ spam như "free" hay "win" không đủ để kéo xác suất qua 0,5 khi phần còn lại của tin (class, coffee, treat, match, tomorrow) toàn từ đời thường. Đây không đúng với giả thiết ban đầu của mình rằng BoW sẽ nhầm ở nhóm này.

Kết luận: điểm yếu của cả hai model không nằm ở các từ "free/win" mà ở spam viết như tin bình thường. Cần thêm dữ liệu huấn luyện kiểu spam giọng tự nhiên, hoặc dùng LLM/router cho các tin có p ở vùng lưỡng lự.*

## 6. Phương án cho sản phẩm thật

*Yêu cầu: 10 triệu SMS/ngày (khoảng 116 tin/giây trung bình), p95 < 50 ms/tin.

Chọn: Encoder (MiniLM) + LogReg (3b). Số liệu: p95 ≈ 43 ms (đạt SLA), F1 SUB 40 là 0,974, cao nhất trong các cách đạt SLA. So với BoW + MLP (F1 SUB 0,889, p95 ≈ 0,25 ms), nó giảm lỗi khoảng 8 điểm nhưng chậm hơn khoảng 170 lần và cần nạp model MiniLM.

Đánh đổi chấp nhận: p95 ≈ 43 ms sát vạch 50 ms, nên chỉ cần máy chậm đi một chút là vượt SLA. Để an toàn có thể dùng GPU hoặc thêm máy chủ. Nếu cần độ trễ cực thấp thì lùi về BoW + MLP và chấp nhận nhiều lỗi hơn.

Vì sao không dùng LLM cho toàn bộ: theo phép tính tay, 1 triệu tin mất 10,50 USD/ngày (100 token vào và 1 token ra mỗi tin, giá 0,10 và 0,50 USD cho mỗi triệu token), nên 10 triệu tin mất khoảng 105 USD/ngày, tức khoảng 3.150 USD/tháng. LLM còn không đạt SLA vì p95 của LLM chạy trong thí nghiệm lên tới hàng nghìn ms.

Hệ thống lai (router): dùng Encoder + LogReg cho mọi tin. Nếu xác suất spam p < 0,3 hoặc p > 0,7 thì quyết định ngay. Chỉ tin có p trong khoảng 0,3 đến 0,7 mới chuyển cho LLM, nên LLM chỉ xử lý một phần nhỏ lượng tin. Cách này giữ chi phí và độ trễ thấp, mà vẫn dùng LLM cho các tin khó như spam giọng tự nhiên (nhóm làm mọi model sai trong bộ 50 tin). Tin chuyển cho LLM nên xử lý bất đồng bộ để không phá SLA.*

## 7. AI log và điều đã học

*AI log
Phần 2 (Deep Learning). Mình nhờ AI viết hàm build_mlp và vòng lặp huấn luyện PyTorch cho mạng MLP một lớp ẩn. AI đưa lại đoạn code dùng nn.Sequential với Linear, ReLU, Linear, và các bước opt.zero_grad(), loss.backward(), opt.step(). Mình chạy bộ kiểm tra thì đạt, rồi xem đường loss và F1 để chắc mạng học đúng. Mình cũng tự thử bỏ ReLU và thấy F1 giảm, nhờ đó giải thích được vì sao cần hàm phi tuyến.

Phần 3 (Embedding). Mình nhờ AI viết cosine_similarity, top_k_similar và knn_predict. AI trả về code chuẩn hoá vector, tính điểm bằng phép nhân ma trận, sắp xếp lấy k tin gần nhất và bỏ phiếu theo đa số. Mình chạy bộ kiểm tra, xem kết quả tìm kiếm theo nghĩa, và hiểu vì sao BoW không thấy được một số từ khoá của câu truy vấn.

Phần 4 (LLM). Mình nhờ AI viết các hàm call_openai, build_classification_prompt, parse_label, llm_classify_batch và cấu hình nhà cung cấp. AI viết code gọi API, prompt zero-shot và few-shot, hàm đọc nhãn (có xử lý trường hợp “not spam”), cơ chế giãn tốc độ và thử lại khi gặp lỗi 429, rồi cấu hình Ollama. Trong lúc làm mình gặp lỗi 429 của Gemini, NameError do thiếu một dòng khởi tạo, và lỗi “thinking level” không được hỗ trợ. Mình sửa theo hướng dẫn rồi chuyển sang chạy Ollama cục bộ cho nhanh và không tốn tiền. Mình kiểm tra bằng bộ kiểm tra của notebook và đếm số nhãn không hợp lệ.

Phần 5 (Tổng hợp). Mình nhờ AI giúp đọc biểu đồ Pareto, bảng kết quả 50 case khó và soạn bản nháp cho Câu hỏi 5 và 6. AI đưa bản nháp và nhận xét rằng LLM nhỏ gần như luôn đoán “ham”. Mình đối chiếu từng con số với bảng và biểu đồ trong notebook rồi viết lại bằng lời của mình.

Thử thách cá nhân #5. Mình tự chạy code, đối chiếu kết quả (mỗi model đúng 2/4) và viết lời giải thích cho khớp số liệu thật. 

Phản ánh cuối bài. Mình nhờ AI nháp 3 câu phản ánh. Mình đối chiếu với bảng so sánh và chỉnh lại ý theo kết quả thật của mình, đặc biệt ở phần dự đoán sai nhiều nhất.

Điều mình hiểu ra hôm nay. Model tốt trên test thường chưa chắc bền khi gặp dữ liệu khó, F1 của Encoder + LogReg giảm từ 0,974 xuống 0,583 trên 50 tin khó. Chọn model là chọn đánh đổi giữa chất lượng, độ trễ và chi phí, và biểu đồ Pareto giúp loại ngay các cách bị lấn át. Model quá nhỏ cũng làm phép so sánh mất ý nghĩa, như LLM 0,5B luôn đoán “ham” nên không đánh giá được prompt phòng thủ.*
