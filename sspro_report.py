# Modified from https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/main/eval_screenspot_pro_parallel.py

import itertools

def zoomin_pred_map(results):
    # map the zoomin predict result back to the point in the original image
    for example in results:
        pred = example['pred']
        if pred is None:
            continue
        x, y = pred
        crop_x1, crop_y1, crop_x2, crop_y2 = example['crop_bbox'] # (x1, y1, x2, y2)
        pred = (crop_x1 + x, crop_y1 + y)
        example['pred'] = pred
        # update back the image size info
        example['crop_image_size'] = example['image_size']
        example['image_size'] = example['orig_image_size']


def judge_correctness(results):
    for example in results:
        pred = example['pred']
        if pred is None:
            example['correctness'] = "wrong_format"
            continue
        bbox = example['bbox']
        x, y = pred
        x1, y1, x2, y2 = bbox
        if (x1 <= x <= x2) and (y1 <= y <= y2):
            example['correctness'] = "correct"
        else:
            example['correctness'] = "wrong"


def ssv2_judge_correctness(results):
    for example in results:
        pred = example['pred']
        if pred is None:
            example['correctness'] = "wrong_format"
            continue
        bbox = example['bbox'] # ScreenSpot-V2 bbox is (x, y, w, h) format, need to convert to (x1, y1, x2, y2) format for the judge_correctness function
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        x, y = pred
        x1, y1, x2, y2 = bbox
        if (x1 <= x <= x2) and (y1 <= y <= y2):
            example['correctness'] = "correct"
        else:
            example['correctness'] = "wrong"


def collect_results_to_eval(results, platform=None, group=None, application=None, language=None, gt_type=None, instruction_style=None, ui_type=None):
    # ... (unchanged from original)
    filtered_results = []
    for sample in results:
        if (platform is None or sample.get("platform") == platform) and \
           (group is None or sample.get("group") == group) and \
           (application is None or sample.get("application") == application) and \
           (language is None or sample.get("language") == language) and \
           (gt_type is None or sample.get("gt_type") == gt_type) and \
           (instruction_style is None or sample.get("instruction_style") == instruction_style) and \
           (ui_type is None or sample.get("ui_type") == ui_type):
            filtered_results.append(sample)
    return filtered_results


def make_combinations(results, platform=False, group=None, application=False, language=False, gt_type=False, instruction_style=False, ui_type=False):
    # ... (unchanged from original)
    unique_values = {
        "platform": set(),
        "group": set(),
        "application": set(),
        "language": set(),
        "gt_type": set(),
        "instruction_style": set(),
        "ui_type": set(),
    }
    for sample in results:
        if platform:
            unique_values["platform"].add(sample.get("platform"))
        if group:
            unique_values["group"].add(sample.get("group"))
        if application:
            unique_values["application"].add(sample.get("application"))
        if language:
            unique_values["language"].add(sample.get("language"))
        if gt_type:
            unique_values["gt_type"].add(sample.get("gt_type"))
        if instruction_style:
            unique_values["instruction_style"].add(sample.get("instruction_style"))
        if ui_type:
            unique_values["ui_type"].add(sample.get("ui_type"))
    filtered_values = {key: list(value) for key, value in unique_values.items() if value}
    if not filtered_values:
        return []
    attribute_combinations = list(itertools.product(*filtered_values.values()))
    combinations = [dict(zip(filtered_values.keys(), combination)) for combination in attribute_combinations]
    return combinations


def calc_metric_for_result_list(results):
    # ... (unchanged from original)
    num_total = len(results)
    correct_num = sum(1 for res in results if res["correctness"] == "correct")
    wrong_format_num = sum(1 for res in results if res["correctness"] == "wrong_format")
    text_results = collect_results_to_eval(results, ui_type="text")
    icon_results = collect_results_to_eval(results, ui_type="icon")
    text_correct = sum(1 for res in text_results if res["correctness"] == "correct")
    text_total = len(text_results)
    icon_correct = sum(1 for res in icon_results if res["correctness"] == "correct")
    icon_total = len(icon_results)
    metrics = {
        "num_correct_action": correct_num,
        "num_total": num_total,
        "wrong_format_num": wrong_format_num,
        "action_acc": correct_num / num_total if num_total > 0 else 0,
        "text_acc": text_correct / text_total if text_total > 0 else 0,
        "icon_acc": icon_correct / icon_total if icon_total > 0 else 0
    }
    return metrics


def eval_sample_positive_gt(sample, response):
    # ... (unchanged from original)
    bbox = sample["bbox"]
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    img_size = sample["img_size"]
    bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    click_point = response["point"]
    print(click_point)
    if click_point is None:
        return "wrong_format"
    if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
        return "correct"
    else:
        return "wrong"
    

def evaluate_leaderboard_detailed_style(results):
    # ... (unchanged from original)
    combinations = make_combinations(results, application=True)
    evaluation_result = {}
    for combo in combinations:
        application = combo.get("application")
        filtered_results = collect_results_to_eval(results, application=application)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"app:{application}"
        evaluation_result[key] = metrics
    return evaluation_result


def evaluate_leaderboard_simple_style(results):
    # ... (unchanged from original)
    combinations = make_combinations(results, group=True)
    evaluation_result = {}
    for combo in combinations:
        group = combo.get("group")
        filtered_results = collect_results_to_eval(results, group=group)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"group:{group}"
        evaluation_result[key] = metrics
    return evaluation_result

# ScreenSpot-V2 only need this simple style report, with report dim by "platform" instead of "group"
# Then "text" and "icon" are self-contained in report by "text_acc" and "icon_acc" respectively. "avg" will be "action_acc" in the report
def ssv2_evaluate_leaderboard_simple_style(results):
    # ... (unchanged from original)
    combinations = make_combinations(results, platform=True)
    evaluation_result = {}
    for combo in combinations:
        platform = combo.get("platform")
        filtered_results = collect_results_to_eval(results, platform=platform)
        metrics = calc_metric_for_result_list(filtered_results)
        if metrics['num_total'] == 0:
            continue
        key = f"platform:{platform}"
        evaluation_result[key] = metrics
    return evaluation_result

def evaluate_overall(results):
    # ... (unchanged from original)
    metrics = calc_metric_for_result_list(results)
    return metrics


def sspro_evaluate(results):
    judge_correctness(results)
    leaderboard_simple_style = evaluate_leaderboard_simple_style(results)
    leaderboard_detailed_style = evaluate_leaderboard_detailed_style(results)
    overall = evaluate_overall(results)
    # put overall first for convenience of viewing
    result_report = {
        "metrics": {
            "overall": overall,
            "leaderboard_simple_style": leaderboard_simple_style,
            "leaderboard_detailed_style": leaderboard_detailed_style,
        },
        "details": results
    }
    return result_report


def ssv2_evaluate(results):
    ssv2_judge_correctness(results)
    leaderboard_simple_style = ssv2_evaluate_leaderboard_simple_style(results)
    overall = evaluate_overall(results)
    # put overall first for convenience of viewing
    result_report = {
        "metrics": {
            "overall": overall,
            "leaderboard_simple_style": leaderboard_simple_style,
        },
        "details": results
    }
    return result_report


def sspro_zoomin_evaluate(results):
    zoomin_pred_map(results)
    return sspro_evaluate(results)