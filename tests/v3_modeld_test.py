"""Compact BPE of the EG2 daemon: merge order, byte fallback, special tokens and the 512 cut, on a toy vocabulary."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "template/.claude/scripts"))
try:
    import numpy  # noqa: F401
except ImportError:
    numpy = None
import memrove_modeld as md


@unittest.skipUnless(numpy, "numpy not installed")
class CompactBpeTest(unittest.TestCase):
    def setUp(self):
        toks = ["<pad>", "<eos>", "<bos>", "<unk>", "a", "b", "▁", "ab", "▁ab", "aab"] + ["<0x%02X>" % i for i in range(256)]
        vocab = {t: i for i, t in enumerate(toks)}
        tj = {"model": {"vocab": vocab, "merges": [["a", "b"], ["▁", "ab"], ["a", "ab"]]}, "added_tokens": [{"id": 3, "content": "<unk>"}]}
        self.tmp = tempfile.TemporaryDirectory()
        src, npz = Path(self.tmp.name, "t.json"), Path(self.tmp.name, "t.npz")
        src.write_text(json.dumps(tj))
        md.CompactBPE.build(str(src), str(npz))
        self.bpe, self.v = md.CompactBPE(str(npz)), vocab

    def tearDown(self):
        self.tmp.cleanup()

    def test_merges_by_rank_with_bos_eos(self):
        v = self.v
        self.assertEqual(self.bpe.encode("ab"), [2, v["ab"], 1])
        self.assertEqual(self.bpe.encode("a ab"), [2, v["a"], v["▁ab"], 1])
        self.assertEqual(self.bpe.encode("aab"), [2, v["aab"], 1])  # (a,b) at rank 0 first, then (a,ab)

    def test_byte_fallback_special_and_cut(self):
        v = self.v
        self.assertEqual(self.bpe.encode("é"), [2, v["<0xC3>"], v["<0xA9>"], 1])
        self.assertEqual(self.bpe.encode("a<unk>b"), [2, v["a"], 3, v["b"], 1])
        out = self.bpe.encode("b" * 700)
        self.assertEqual((len(out), out[0], out[-1]), (512, 2, 1))


if __name__ == "__main__":
    unittest.main()
