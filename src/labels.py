from typing import Dict, List, Tuple, Optional
import re


def _read_excel_pairs(path: str) -> List[Tuple[str, Optional[str]]]:
	"""从 Excel 读取键值对列表（尽量容错）。

	规则：
	- 取第一个工作表；
	- 每行取前两个非空字符串单元格作为 (key, value)，若只有一个则 (key, None)；
	"""
	import openpyxl  # 仅在使用时导入，避免编译期依赖
	wb = openpyxl.load_workbook(path, data_only=True)
	ws = wb.worksheets[0]
	pairs: List[Tuple[str, Optional[str]]] = []
	rec_id_pattern = re.compile(r"^\d{8}_s\d{3}_t\d{3}$", re.IGNORECASE)
	header_pattern = re.compile(r"^(class( code)?|label|alias|规范名|别名|类型|类别|name|code)$", re.IGNORECASE)
	noise_word_pattern = re.compile(r"(file(name)?|record|patient|start|end|onset|offset|duration|confidence|备注|说明|sheet|table|index|序号|编号|id)$", re.IGNORECASE)
	for row in ws.iter_rows(values_only=True):
		cells = [c for c in row if c is not None]
		cells = [str(c).strip() for c in cells if str(c).strip()]
		if not cells:
			continue
		k0 = cells[0]
		v0 = cells[1] if len(cells) > 1 else None
		# 跳过表头/噪声关键字
		if header_pattern.match(k0) or (v0 is not None and header_pattern.match(v0)):
			continue
		if noise_word_pattern.search(k0):
			continue
		# 过滤记录ID样式或纯数字/时间戳样式的行
		if rec_id_pattern.match(k0):
			continue
		if k0.replace(".", "", 1).isdigit():
			continue
		# 过滤别名列为记录ID/纯数字的情况
		if v0 is not None:
			if rec_id_pattern.match(v0):
				v0 = None
			elif v0.replace(".", "", 1).isdigit():
				v0 = None
			elif noise_word_pattern.search(v0):
				v0 = None
		if v0 is None:
			pairs.append((k0, None))
		else:
			pairs.append((k0, v0))
	return pairs


def build_labels_from_excels(
	types_xlsx: Optional[str],
	periods_xlsx: Optional[str],
	background: str = "bckg",
) -> Tuple[List[str], Dict[str, str]]:
	"""从 Excel 表构建 label 名单与别名映射。

	返回：
	- label_names: 有序标签列表（含 background 置于首位）
	- aliases: 别名到规范名的字典（大小写不敏感，内部统一小写）
	"""
	canon: Dict[str, None] = {}
	alias_map: Dict[str, str] = {}

	# 仅从“发作类型表”定义类别集合
	if types_xlsx:
		for k, v in _read_excel_pairs(types_xlsx):
			k_norm = k.strip()
			if not k_norm:
				continue
			canon.setdefault(k_norm, None)
			# 自身别名（小写）
			alias_map.setdefault(k_norm.lower(), k_norm)
			# 第二列作为别名映射到规范名
			if v is not None and v.strip() and v.strip().lower() != k_norm.lower():
				alias_map[v.strip().lower()] = k_norm

	# “时段类型表”仅用于增加别名映射，不新增类别
	if periods_xlsx and canon:
		canon_l2c = {c.lower(): c for c in canon.keys()}
		for k, v in _read_excel_pairs(periods_xlsx):
			k_s = k.strip() if k else ""
			v_s = v.strip() if (v is not None) else ""
			# 若第二列（别名→规范名）已在类别集合中，则添加别名映射
			if v_s and v_s.lower() in canon_l2c:
				alias_map[k_s.lower()] = canon_l2c[v_s.lower()]
			# 或者第一列即为规范名，则接受第二列为其别名
			elif k_s and k_s.lower() in canon_l2c and v_s:
				alias_map[v_s.lower()] = canon_l2c[k_s.lower()]

	# 确保 background 在首位
	labels = [background]
	for name in canon.keys():
		if name.lower() == background.lower():
			continue
		labels.append(name)
	return labels, alias_map


def write_labels_json(out_path: str, label_names: List[str], aliases: Dict[str, str], background: str) -> None:
	import json, os
	os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		json.dump({
			"background": background,
			"label_names": label_names,
			"aliases": aliases,
		}, f, ensure_ascii=False, indent=2)


