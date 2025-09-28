# tasks.py
from crewai import Task
from datetime import datetime

class StockAnalysisTasks():
    def news_collecting(self, agent, symbol):
        return Task(
            description=f"""
                Tìm kiếm và tóm tắt 3 bài báo có ảnh hưởng lớn nhất đến công ty {symbol} và thị trường chứng khoán Việt Nam
                trong vòng 3 tháng tính đến ngày hiện tại ({datetime.now().strftime('%Y-%m-%d')}).

                Quy trình thực hiện:
                1. Sử dụng công cụ `SerperDevTool` để tìm các bài báo liên quan đến công ty {symbol}, ngành nghề của công ty, và tin tức vĩ mô, chính sách kinh tế Việt Nam (ví dụ: dự báo tăng trưởng, lãi suất, thuế, đầu tư công, tỉ giá hối đoái...). Ưu tiên kết quả từ các nguồn uy tín.
                2. Chọn 3 bài báo tiêu biểu nhất dựa trên mức độ ảnh hưởng, độ tin cậy và mức độ liên quan đến thị trường chứng khoán Việt Nam. KHÔNG chọn các bài viết từ nguồn vietstock.vn.
                3. Sử dụng công cụ `ScrapeWebsiteTool` để thu thập nội dung chi tiết của từng bài báo đã chọn.
                4. Tóm tắt nội dung chính của từng bài trong 3–5 câu, tập trung vào tác động đến thị trường chứng khoán và công ty {symbol}.

                Giới hạn:
                - Chỉ chọn bài viết có ngày đăng trong vòng 3 tháng trở lại.
                - KHÔNG lựa chọn hay sử dụng các bài viết từ nguồn vietstock.vn.
                - Chỉ chọn bài có tiêu đề và nội dung thể hiện rõ ảnh hưởng đến thị trường tài chính/chứng khoán Việt Nam.
                - KHÔNG được tự tạo ra nội dung. Câu trả lời phải dựa trên dữ liệu trích xuất từ các công cụ.
            """,
            expected_output="""
                Một danh sách dạng markdown gồm 3 mục, mỗi mục bao gồm:
                - Tiêu đề bài báo (in đậm)
                - Ngày đăng
                - Nguồn (kèm liên kết URL)
                - Tóm tắt 3–5 câu bằng tiếng Việt nêu rõ nội dung chính và ảnh hưởng đến thị trường
            """,
            agent=agent,
            async_execution=True
        )

    def fundamental_analysis(self, agent, symbol):
        return Task(
            description=f"""
                Phân tích cơ bản mã cổ phiếu {symbol} dựa trên các chỉ số tài chính để đánh giá mức định giá và sức khỏe doanh nghiệp.

                Quy trình thực hiện:
                1. Sử dụng công cụ `Công cụ tra cứu dữ liệu cổ phiếu phục vụ phân tích cơ bản` để thu thập các chỉ số: P/E, P/B, ROE, D/E, EPS, EV/EBITDA, tăng trưởng doanh thu/lợi nhuận, và biên lợi nhuận.
                2. Xác định cổ phiếu thuộc ngành nào từ kết quả của công cụ.
                3. Đọc và sử dụng file `knowledge/PE_PB_industry_average.json` để so sánh P/E và P/B của cổ phiếu với trung bình ngành.
                   Nếu ngành chưa có dữ liệu, sử dụng ngành gần nhất tương đương và ghi chú lại.
                4. Phân tích các chỉ số còn lại để đánh giá hiệu suất hoạt động và mức độ rủi ro tài chính.

                Ngày thực hiện: {datetime.now().strftime('%Y-%m-%d')}
            """,
            expected_output="""
                Một báo cáo bằng tiếng Việt, dài 2–3 đoạn, bao gồm:
                - Nhận định định giá: rẻ / đắt / hợp lý dựa trên so sánh P/E, P/B với ngành.
                - Đánh giá hiệu quả hoạt động tài chính: liệu các chỉ số ROE, biên lợi nhuận, tăng trưởng có tốt không.
                - Nhận xét tổng thể về sức khỏe tài chính và triển vọng kinh doanh.
            """,
            agent=agent,
            async_execution=True
        )

    def technical_analysis(self, agent, symbol):
        return Task(
            description=f"""
                Đánh giá xu hướng giá của cổ phiếu {symbol} trong ngày {datetime.now().strftime('%Y-%m-%d')} thông qua phân tích kỹ thuật.

                Quy trình thực hiện:
                1. Sử dụng công cụ `Công cụ tra cứu dữ liệu cổ phiếu phục vụ phân tích kĩ thuật` để thu thập các chỉ báo: SMA, EMA, RSI, MACD, khối lượng giao dịch, OBV và xác định vùng hỗ trợ/kháng cự.
                2. Dựa trên kết quả trả về từ tool, xác định xu hướng hiện tại (tăng/giảm/tích lũy).
                3. Đưa ra đánh giá động lượng và khả năng xuất hiện tín hiệu giao dịch.
            """,
            expected_output="""
                Một báo cáo bằng tiếng Việt, dài 2–3 đoạn, bao gồm:
                - Nhận định xu hướng hiện tại (tăng/giảm/đi ngang) dựa trên các đường MA.
                - Phân tích tín hiệu kỹ thuật (ví dụ: RSI quá mua/quá bán, giao cắt MACD...).
                - Đánh giá tổng thể về thời điểm mua bán từ góc nhìn kỹ thuật, đề cập tới các vùng hỗ trợ/kháng cự quan trọng.
            """,
            agent=agent,
            async_execution=True
        )

    def investment_decision(self, agent, symbol, context):
        return Task(
            description=f"""
                Tổng hợp thông tin từ các phân tích vĩ mô và kỹ thuật đã thực hiện để đưa ra một chiến lược đầu tư cho mã {symbol}.
                LƯU Ý: Phân tích này KHÔNG có thông tin chi tiết từ báo cáo tài chính PDF.
                Bạn phải chấm điểm từng yếu tố (vĩ mô, kỹ thuật) và phân tích cơ bản (từ API) trên thang điểm 10 và giải thích lý do.
                Dựa trên điểm trung bình, đưa ra khuyến nghị rõ ràng: MUA, BÁN, hoặc GIỮ.
                - Điểm >= 7.5: MUA
                - 4.0 <= Điểm < 7.5: GIỮ
                - Điểm < 4.0: BÁN
                Nếu là GIỮ, hãy đề xuất vùng giá mua và bán tiềm năng.
            """,
            expected_output=f"""
                Một báo cáo phân tích đầy đủ bằng tiếng Việt, có cấu trúc markdown chuẩn như sau:
                ---
                ## BÁO CÁO PHÂN TÍCH CỔ PHIẾU {symbol}
                **Ngày báo cáo:** {datetime.now().strftime('%Y-%m-%d')}

                ### 1. Phân tích Vĩ mô & Tin tức
                (Tóm tắt các tin tức quan trọng và tác động của chúng)
                
                **Điểm:** [Điểm/10]
                **Lý do:** (Giải thích ngắn gọn tại sao lại chấm điểm như vậy)

                ### 2. Phân tích Cơ bản
                (Tóm tắt định giá, sức khỏe tài chính, và hiệu quả hoạt động, cạnh tranh)
                
                **Điểm:** [Điểm/10]
                **Lý do:** (Giải thích ngắn gọn)

                ### 3. Phân tích Kỹ thuật
                (Tóm tắt xu hướng, động lượng và các tín hiệu kỹ thuật)

                **Điểm:** [Điểm/10]
                **Lý do:** (Giải thích ngắn gọn)

                ### 4. Kết luận và Khuyến nghị
                **Điểm trung bình:** [Điểm trung bình/10]
                
                **Khuyến nghị:** **[MUA/GIỮ/BÁN]**
                
                **(Nếu GIỮ, thêm phần này)**
                **Vùng giá tham khảo:**
                - **MUA khi giá điều chỉnh về vùng:** [Giá mục tiêu mua]
                - **BÁN khi giá đạt mục tiêu tại:** [Giá mục tiêu bán]
                ---
            """,
            agent=agent,
            context=context
        )

    def analyze_financial_report(self, agent, file_path, company_ticker=None):
        description = f"""
            Phân tích chuyên sâu báo cáo tài chính từ file PDF tại đường dẫn: '{file_path}'.
            Quy trình thực hiện:
            1. Sử dụng công cụ 'Công cụ trích xuất văn bản từ file PDF' để đọc toàn bộ nội dung của file.
            2. Đọc nội dung file text đã được trích xuất.
            3. Dựa vào nội dung, tóm tắt những điểm tài chính quan trọng nhất trong báo cáo.
            4. Soạn thảo một báo cáo tổng kết súc tích.
        """
        if company_ticker:
            description += f"\n   Lưu ý: Báo cáo này của công ty có mã cổ phiếu là {company_ticker}."
        
        return Task(
            description=description,
            expected_output="""
                Một báo cáo markdown chuyên nghiệp tóm tắt báo cáo tài chính, bao gồm các mục sau:
                - **Tổng quan kết quả kinh doanh:** Doanh thu, lợi nhuận và so sánh với cùng kỳ.
                - **Phân tích biên lợi nhuận:** Đánh giá hiệu quả hoạt động.
                - **Sức khỏe tài chính:** Phân tích bảng cân đối kế toán, nợ vay, vốn chủ sở hữu.
                - **Dòng tiền:** Đánh giá chất lượng dòng tiền.
                - **Kết luận:** Những điểm nhấn và rủi ro chính cần lưu ý.
            """,
            agent=agent,
            async_execution=False 
        )
    
    def competitor_analysis(self, agent, symbol):
        return Task(
            description=f"""
                Phân tích đối thủ cạnh tranh cho công ty có mã cổ phiếu {symbol}.
                Quy trình thực hiện:
                1. Sử dụng công cụ 'Search the internet with Serper' để tìm ra 2-3 đối thủ cạnh tranh chính, cùng ngành, đã niêm yết trên sàn chứng khoán Việt Nam. Truy vấn tìm kiếm nên là "đối thủ cạnh tranh của {symbol} trong ngành công nghệ".
                2. Với mỗi đối thủ tìm được, lấy mã cổ phiếu của họ.
                3. Sử dụng công cụ 'Công cụ tra cứu dữ liệu cổ phiếu phục vụ phân tích cơ bản' cho từng mã cổ phiếu đối thủ để thu thập các chỉ số tài chính quan trọng (P/E, P/B, ROE, Biên lợi nhuận).
                4. Tạo một bảng so sánh các chỉ số này giữa {symbol} và các đối thủ.
                5. Dựa trên bảng so sánh, đưa ra nhận định ngắn gọn về vị thế cạnh tranh của {symbol} so với các đối thủ (ví dụ: định giá cao hơn/thấp hơn, hiệu quả hoạt động tốt hơn/kém hơn).
            """,
            expected_output=f"""
                Một báo cáo markdown về phân tích cạnh tranh, bao gồm:
                - Danh sách 2-3 đối thủ cạnh tranh chính và mã cổ phiếu của họ.
                - Một bảng so sánh các chỉ số tài chính (P/E, P/B, ROE, Biên lợi nhuận) giữa {symbol} và các đối thủ.
                - Nhận định cuối cùng về lợi thế cạnh tranh của {symbol} dựa trên các số liệu so sánh.
            """,
            agent=agent,
            async_execution=False
        )

    def comprehensive_stock_analysis(self, agent, symbol, context):
        return Task(
            description=f"""
                Đưa ra một phân tích đầu tư TOÀN DIỆN và SÂU SẮC cho mã cổ phiếu {symbol} bằng cách kết hợp thông tin từ TẤT CẢ các nguồn được cung cấp:
                1. Phân tích tin tức vĩ mô.
                2. Phân tích kỹ thuật.
                3. Phân tích chi tiết từ Báo cáo tài chính (PDF).
                4. Phân tích so sánh với đối thủ cạnh tranh.

                Quy trình của bạn:
                1. Đọc và hiểu rõ kết quả từ tất cả các bản phân tích trước đó.
                2. Tìm ra sự liên kết giữa các phân tích. Ví dụ: "Kết quả kinh doanh vượt trội trong BCTC (phân tích PDF) và cao hơn hẳn so với đối thủ (phân tích cạnh tranh) đã được phản ánh vào giá cổ phiếu..."
                3. Chấm điểm từng yếu tố (Vĩ mô, Cơ bản & Cạnh tranh, Kỹ thuật) trên thang điểm 10.
                4. Đưa ra khuyến nghị cuối cùng (MUA/BÁN/GIỮ) và luận điểm đầu tư sắc bén.
            """,
            expected_output="""
                Một báo cáo đầu tư TOÀN DIỆN, bao gồm:
                - **Luận điểm đầu tư:** Đoạn văn 3-5 câu tóm tắt lý do chính.
                - **Phân tích tổng hợp:**
                  - **Vĩ mô & Tin tức:** Tóm tắt và chấm điểm.
                  - **Cơ bản (BCTC & Cạnh tranh):** Tổng hợp từ PDF và so sánh đối thủ, sau đó chấm điểm.
                  - **Kỹ thuật:** Tóm tắt xu hướng và chấm điểm.
                - **Kết luận và Khuyến nghị:** Điểm trung bình, Khuyến nghị, và Vùng giá tham khảo.
            """,
            agent=agent,
            context=context
        )
