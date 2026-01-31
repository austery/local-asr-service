import pytest
from src.adapters.text import clean_sensevoice_tags

class TestTextAdapter:
    """
    测试 src/adapters/text.py 中的文本清洗逻辑
    """

    def test_basic_cleaning(self):
        """测试基础的标签清洗功能"""
        raw_text = "<|zh|><|NEUTRAL|>你好，世界！<|Speech|>"
        expected = "你好，世界！"
        assert clean_sensevoice_tags(raw_text) == expected

    def test_empty_input(self):
        """测试空输入"""
        assert clean_sensevoice_tags(None) == ""
        assert clean_sensevoice_tags("") == ""

    def test_no_tags(self):
        """测试不含标签的普通文本"""
        raw_text = "Hello World"
        assert clean_sensevoice_tags(raw_text) == "Hello World"

    def test_whitespace_normalization(self):
        """测试清洗后多余空格的合并"""
        # 假设标签在中间，去掉后可能留下双空格
        raw_text = "Hello <|tag|> World"
        expected = "Hello World"
        assert clean_sensevoice_tags(raw_text) == expected

    def test_clean_tags_disabled(self):
        """测试禁用清洗功能"""
        raw_text = "<|zh|>Test"
        # 当 clean_tags=False 时，应原样返回
        assert clean_sensevoice_tags(raw_text, clean_tags=False) == raw_text

    def test_spaced_tags(self):
        """测试带空格的 SenseVoice 标签 (e.g. < | zh | >)"""
        raw_text = "< | zh | > < | NEUTRAL | > < | S pe ech | > < | withi tn | >哦，，智能清洗其实是在一个源头做的。"
        expected = "哦，智能清洗其实是在一个源头做的。"
        assert clean_sensevoice_tags(raw_text) == expected

    def test_punctuation_normalization(self):
        """测试标点符号规范化 (去除重复标点)"""
        raw_text = "你好，，世界。。Hello,, World.."
        expected = "你好，世界。Hello, World."
        assert clean_sensevoice_tags(raw_text) == expected

