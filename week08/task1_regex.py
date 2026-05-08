"""Task 1: 用 regex 擷取日期、電話、Email"""
import re

TEXT = """
王小明於2025年8月15日到台北市中山區松江路123號拜訪陳小姐，
聯絡電話為(02) 2345-6789，Email: test@example.com。
"""

PAT_DATE = re.compile(r'(\d{4})年(\d{1,2})月(\d{1,2})日')
PAT_PHONE = re.compile(r'\(\d{2}\)\s*\d{4}-\d{4}')
PAT_EMAIL = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')

print("=" * 50)
print("Task 1: Regex Extraction")
print("=" * 50)
print("原文：", TEXT.strip())
print("-" * 50)
print("Dates :", PAT_DATE.findall(TEXT))
print("Phones:", PAT_PHONE.findall(TEXT))
print("Emails:", PAT_EMAIL.findall(TEXT))
