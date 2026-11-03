import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_corpus", ROOT/"scripts/audit_corpus.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class CorpusAuditTests(unittest.TestCase):
    def test_page_flags_and_source(self):
        row = audit.inspect_markdown("# T\n> Source: `test.pdf`\n## Page 1\n正文\n## Page 3\n[OCR_REQUIRED]\n## Page 3\n")
        self.assertEqual(row["page_markers"], 3)
        self.assertEqual(row["missing_page_numbers"], [2])
        self.assertEqual(row["duplicate_page_numbers"], [3])
        self.assertEqual(row["explicit_ocr_failure_or_required_markers"], 1)
        self.assertEqual(row["source_header"], "test.pdf")

    def test_zero_and_out_of_order_pages(self):
        row = audit.inspect_markdown("## Page 2\na\n## Page 0\nb\n## Page 1\nc")
        self.assertTrue(row["out_of_order_page_numbers"])
        self.assertEqual(row["nonpositive_page_numbers"], [0])

    def test_no_pages_does_not_certify_good_document(self):
        row = audit.inspect_markdown("没有页码也没有Source")
        self.assertEqual(row["page_markers"], 0)
        self.assertIsNone(row["source_header"])

    def test_unicode_normalization(self):
        self.assertEqual(audit.normalized_name("folder/Древнеи\u0306.pdf"), audit.normalized_name("Древней.pdf"))

    def test_missing_and_ambiguous_pdf_mapping(self):
        work = ROOT/".local"
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as tmp:
            folder = Path(tmp)
            md, pdf = folder/"md", folder/"pdf"
            md.mkdir(); pdf.mkdir()
            (pdf/"a").mkdir(); (pdf/"b").mkdir()
            # Header matching only: placeholders are not opened as PDFs.
            for path in (pdf/"a/test.pdf", pdf/"b/test.pdf", pdf/"other.pdf"):
                path.write_bytes(b"placeholder")
            (md/"test.md").write_text("> Source: `test.pdf`\n## Page 1\ntext", encoding="utf-8")
            result = audit.inventory(md, pdf)
            self.assertEqual(result["summary"]["pdf_mapping_counts"], {"ambiguous": 1})
            self.assertEqual(result["summary"]["pdfs_not_uniquely_matched_to_markdown"], 3)
            self.assertFalse(result["safety"]["source_documents_modified"])


if __name__ == "__main__":
    unittest.main()
