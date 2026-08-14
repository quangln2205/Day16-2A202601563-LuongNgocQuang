"""LỚP `citation_checker` — bài giảng Day 16, §11 (Grounding & Citations).

NHIỆM VỤ: chỉ cần MỘT tài liệu gắn nhãn `lookalike` hoặc `outdated` lọt
vào bằng chứng là mô hình neo TOÀN BỘ claim vào đúng tài liệu trông có vẻ
"chính thống" đó — dù mỗi câu được lấy nguyên văn từ một tài liệu khác.
Câu thì thật, trích dẫn thì sai. Đây là kiểu sai nguy hiểm nhất trong RAG
vì báo cáo đọc vào vẫn rất thuyết phục.

TÍN HIỆU (chính xác, không cần đoán):

    claim["text"] KHÔNG khớp NGUYÊN VĂN một DÒNG nào trong
    corpus.get(claim["doc_id"]).body
    nhưng CHÍNH câu đó CÓ trong bằng chứng agent đã quan sát

Chú ý chữ DÒNG: kiểm tra `claim["text"] in doc.body` (cả khối, không
tách dòng) là SAI — scorer chỉ nhận trích dẫn khớp nguyên văn MỘT DÒNG
(xem "ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP" ngay dưới đây). `in doc.body` coi
một câu vắt qua hai dòng là hợp lệ, trong khi scorer thì không — tín
hiệu kiểu đó khiến bạn giữ nguyên một trích dẫn mà scorer vẫn chấm
`HALLUCINATED`.

Vế thứ hai mới là phần quan trọng: nó tách việc của bạn khỏi việc của
`critic` (§2). Câu có trong bằng chứng nhưng gắn sai tài liệu -> GẮN LẠI
(việc của bạn). Câu không có trong bằng chứng nào -> BỊA, để `critic` xoá.
Hai điều kiện loại trừ nhau nên hai lớp không giành điểm của nhau.

ĐƯỢC PHÉP VÀ KHÔNG ĐƯỢC PHÉP:
  * ĐƯỢC: đổi `claim["doc_id"]`, cập nhật `report["citations"]`.
  * KHÔNG: sửa `claim["text"]`. Scorer chỉ cho điểm khi câu là trích dẫn
    nguyên văn của MỘT DÒNG trong tài liệu được trích VÀ đúng là chữ mô
    hình đã viết. Thêm dấu chấm, đổi dấu nháy, "chuẩn hoá" khoảng trắng,
    hay vá lại câu bị cắt bằng nội dung lấy từ corpus đều làm mất cả hai
    điều kiện cùng lúc (đo được: -40 điểm).

CHỈ ĐƯỢC GẮN VÀO TÀI LIỆU ĐÃ QUAN SÁT. Trích một tài liệu mà lượt chạy
chưa từng đọc bị chấm `UNRETRIEVED`. Vì vậy hãy tìm nguồn trong
`ctx.observed_text`, đừng quét cả corpus rồi gắn bừa: điều kiện
`doc.body in ctx.observed_text` nghĩa là "tài liệu này đã về nguyên vẹn
từ một lần fetch sạch" — một đoạn snippet hay một bản bị cắt không tính.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.corpus.get(doc_id) -> Doc | None
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.

Cài đặt:  ReActAgent(..., middleware=[..., CitationChecker(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

from harness.middleware import Middleware


class CitationChecker(Middleware):
    """Trỏ mỗi claim về đúng tài liệu thật sự chứa câu đó."""

    name = "citation_checker"

    def after_agent(self, ctx, report):
        # Get claims from report
        claims = report.get("claims")
        if not isinstance(claims, list) or ctx.corpus is None:
            return report
            
        # Process each claim
        updated_claims = []
        seen_doc_ids = set()
        
        for claim in claims:
            if not isinstance(claim, dict):
                updated_claims.append(claim)
                continue
                
            text = claim.get("text", "")
            doc_id = claim.get("doc_id")
            
            if not text:
                updated_claims.append(claim)
                continue
                
            # Check if the current doc_id is valid
            if doc_id:
                doc = ctx.corpus.get(doc_id)
                if doc and self._is_exact_line_match(text, doc.body):
                    # Current doc_id is correct
                    updated_claims.append(claim)
                    seen_doc_ids.add(doc_id)
                    continue
                    
            # Try to find the correct document
            correct_doc_id = self._find_correct_doc_id(ctx, text)
            if correct_doc_id:
                # Update claim with correct doc_id
                updated_claim = claim.copy()
                updated_claim["doc_id"] = correct_doc_id
                updated_claims.append(updated_claim)
                seen_doc_ids.add(correct_doc_id)
            else:
                # No correct document found, keep original claim
                # But let's also check if the text is in any observed document
                # even if it's not in the current doc_id
                found_doc_id = self._find_any_observed_doc_with_text(ctx, text)
                if found_doc_id:
                    # Update claim with found doc_id
                    updated_claim = claim.copy()
                    updated_claim["doc_id"] = found_doc_id
                    updated_claims.append(updated_claim)
                    seen_doc_ids.add(found_doc_id)
                else:
                    updated_claims.append(claim)
                
        # Update citations
        report["claims"] = updated_claims
        report["citations"] = list(seen_doc_ids)
        
        return report
        
    def _find_any_observed_doc_with_text(self, ctx, text):
        """Find any observed document that contains the text as an exact line."""
        # First check if the text is in any observed document
        for doc in ctx.corpus.docs:
            if doc.body and doc.doc_id in ctx.observed_text:
                # Check if text is an exact line match in this document
                lines = doc.body.split('\n')
                for line in lines:
                    if line.strip() == text.strip():
                        return doc.doc_id
        return None
        
    def _is_exact_line_match(self, text, body):
        """Check if text matches exactly one line in the document body."""
        if not body:
            return False
        lines = body.split('\n')
        # Check if the text is exactly one line (not a substring)
        # We need to check if the text is a complete line in the document
        # The text should be exactly one line, not a substring of a line
        for line in lines:
            if line.strip() == text.strip():
                return True
        return False
        
    def _find_correct_doc_id(self, ctx, text):
        """Find the correct document ID that contains the text as an exact line."""
        # Check all documents for the text
        for doc in ctx.corpus.docs:
            if doc.body and doc.doc_id in ctx.observed_text:
                # Check if text is an exact line match in this document
                lines = doc.body.split('\n')
                for line in lines:
                    if line.strip() == text.strip():
                        return doc.doc_id
        return None
