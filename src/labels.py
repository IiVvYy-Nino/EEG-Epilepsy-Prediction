from typing import Dict, List, Tuple, Optional


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
	for row in ws.iter_rows(values_only=True):
		cells = [c for c in row if c is not None]
		cells = [str(c).strip() for c in cells if str(c).strip()]
		if not cells:
			continue
		if len(cells) == 1:
			pairs.append((cells[0], None))
		else:
			pairs.append((cells[0], cells[1]))
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

	def add_pair(k: str, v: Optional[str]):
		k_norm = k.strip()
		if not k_norm:
			return
		canon.setdefault(k_norm, None)
		if v is not None and v.strip() and v.strip().lower() != k_norm.lower():
			alias_map[v.strip().lower()] = k_norm
		# 自身别名（小写）
		alias_map.setdefault(k_norm.lower(), k_norm)

	if types_xlsx:
		for k, v in _read_excel_pairs(types_xlsx):
			add_pair(k, v)
	if periods_xlsx:
		for k, v in _read_excel_pairs(periods_xlsx):
			add_pair(k, v)

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


