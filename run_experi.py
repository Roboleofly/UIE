import os
import csv
import cv2
from tqdm import tqdm  # 用于进度条显示
from metrics import calculate_uciqe, calculate_uiqm, calculate_entropy  # 你自己的指标实现
from OURS.adaptive_fusion_dev import Enhance_img
from OURS.FusionEnhance import enhance
from UDCP.main import UDCP
from ULAP.main import ULAP
from RGHS.main import RGHS


def Adaptive_enhance(input_path, output_path):
    img = cv2.imread(input_path)
    enhanced_img = Enhance_img(img)
    cv2.imwrite(output_path, enhanced_img)
    return 

def Fusion_enhance(input_path, output_path):
    img = cv2.imread(input_path)
    enhance_img = enhance(img, 5)
    cv2.imwrite(output_path, enhance_img)
    return 

def UDCP_enhance(input_path, output_path):
    img = cv2.imread(input_path)
    enhanced_img = UDCP(img)
    cv2.imwrite(output_path, enhanced_img)
    return

def ULAP_enhance(input_path, output_path):
    img = cv2.imread(input_path)
    enhanced_img = ULAP(img)
    cv2.imwrite(output_path, enhanced_img)
    return

def RGHS_enhance(input_path, output_path):
    img = cv2.imread(input_path)
    enhanced_img = img
    cv2.imwrite(output_path, enhanced_img)
    return

def evaluate_image(image_path, method_name, enhanced_image_path):
    uciqe = calculate_uciqe(enhanced_image_path)
    uiqm = calculate_uiqm(enhanced_image_path)
    ie = calculate_entropy(enhanced_image_path)
    return [image_path, method_name, uciqe, uiqm, ie]

def run_experiment(
    input_dir, output_dir, method_name, enhance_function, result_csv_path
):
    """
    input_dir: 原始图像目录
    output_dir: 保存增强后图像的目录
    method_name: 算法名称
    enhance_function: 实现了增强逻辑的函数
    result_csv_path: 最终记录结果的CSV文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for img_name in tqdm(os.listdir(input_dir), desc=f"Processing with {method_name}"):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        # 使用增强函数处理图像并保存结果
        try:
            enhance_function(input_path, output_path)
            # 计算指标并记录
            result = evaluate_image(input_path, method_name, output_path)
            results.append(result)

        except Exception as e:
            print("发生错误了:", e)

    # 写入CSV文件
    with open(result_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)


def init_csv(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "method_name", "UCIQE", "UIQM", "IE"])

def test(img_path):
    # img = cv2.imread(img_path)
    # enhanced_img = Enhance_img(img)
    uciqe = calculate_uciqe(img_path)
    print(uciqe)
    uiqm = calculate_uiqm(img_path)
    print(uiqm)
    ie = calculate_entropy(img_path)
    print(ie)



# test('/root/workspace/UIE/datasets/UIEB_Dataset/9_img_.png')


if __name__ == '__main__':
    # test 
    # img_path = '/root/workspace/UIE/datasets/UIEB_Dataset/9_img_.png'
    # test(img_path)
    # run_experiment(
    #     input_dir="/media/users/leo/workspace/UIE/datasets/UIEB_Dataset",
    #     output_dir="/media/users/leo/workspace/UIE/expri/UIEB/RGHS",
    #     method_name="RGHS",
    #     enhance_function=RGHS_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )


    # run_experiment(
    #     input_dir="/media/users/leo/workspace/UIE/datasets/OceanDark2_0",
    #     output_dir="/media/users/leo/workspace/UIE/expri/OceanDark2_0/RGHS",
    #     method_name="RGHS",
    #     enhance_function=RGHS_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )

    # run_experiment(
    #     input_dir="/media/users/leo/workspace/UIE/datasets/UIDEF_S",
    #     output_dir="/media/users/leo/workspace/UIE/expri/UIDEF_S/ULAP",
    #     method_name="ULAP",
    #     enhance_function=ULAP_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )

    # run_experiment(
    #     input_dir="/media/users/leo/workspace/UIE/datasets/OceanDark2_0",
    #     output_dir="/media/users/leo/workspace/UIE/expri/OceanDark2_0/ULAP",
    #     method_name="ULAP",
    #     enhance_function=ULAP_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )

    # run_experiment(
    #     input_dir="/media/users/leo/workspace/UIE/datasets/UIEB_Dataset",
    #     output_dir="/media/users/leo/workspace/UIE/expri/UIEB/ULAP",
    #     method_name="ULAP",
    #     enhance_function=ULAP_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )

    # run_experiment(
    #     input_dir="/media/users/leo/workspace/UIE/datasets/UIDEF_S",
    #     output_dir="/media/users/leo/workspace/UIE/expri/UIDEF_S/UDCP",
    #     method_name="UDCP",
    #     enhance_function=UDCP_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )

    # run_experiment(
    #     input_dir="/media/users/leo/workspace/UIE/datasets/UIEB_Dataset",
    #     output_dir="/media/users/leo/workspace/UIE/expri/UIEB/UDCP",
    #     method_name="UDCP",
    #     enhance_function=UDCP_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )

    # run_experiment(
    #     input_dir="/media/users/leo/workspace/UIE/datasets/OceanDark2_0",
    #     output_dir="/media/users/leo/workspace/UIE/expri/OceanDark2_0/UDCP",
    #     method_name="UDCP",
    #     enhance_function=UDCP_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )


    # run_experiment(
    #     input_dir="/media/users/leo/workspace/UIE/datasets/UIDEF_S",
    #     output_dir="/media/users/leo/workspace/UIE/expri/UIDEF_S/OURS",
    #     method_name="OURS",
    #     enhance_function=Adaptive_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )

    # run_experiment(
    #     input_dir="/media/users/leo/workspace/UIE/datasets/UIDEF_S",
    #     output_dir="/media/users/leo/workspace/UIE/expri/UIDEF_S/FUSION2",
    #     method_name="FUSION2",
    #     enhance_function=Fusion_enhance,
    #     result_csv_path="tmp.csv"
    # )

    # run_experiment(
    #     input_dir="/root/workspace/UIE/datasets/UIEB_Dataset",
    #     output_dir="/root/workspace/UIE/expri/UIEB/FUSION2",
    #     method_name="FUSION2",
    #     enhance_function=Fusion_enhance,
    #     result_csv_path="tmp.csv"
    # )
    
    # 跑method 1 
    # run_experiment(
    #     input_dir="/root/workspace/UIE/datasets/UIEB_Dataset",
    #     output_dir="/root/workspace/UIE/expri/UIEB/OURS",
    #     method_name="OURS",
    #     enhance_function=Adaptive_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )

    # run_experiment(
    #     input_dir="/root/workspace/UIE/datasets/UIEB_Dataset",
    #     output_dir="/root/workspace/UIE/expri/UIEB/FUSION2",
    #     method_name="FUSION2",
    #     enhance_function=Fusion_enhance,
    #     result_csv_path="tmp.csv"
    # )

    # 跑 method1
    run_experiment(
        input_dir="/media/users/leo/workspace/UIE/datasets//OceanDark2_0",
        output_dir="/media/users/leo/workspace/UIE/datasets/OURS2",
        method_name="OURS2",
        enhance_function=Adaptive_enhance,
        result_csv_path="evaluation_metrics.csv"
    )

    # run_experiment(
    #     input_dir="/root/workspace/UIE/datasets/OceanDark2_0",
    #     output_dir="/root/workspace/UIE/expri/OceanDark2_0/FUSION2",
    #     method_name="FUSION2",
    #     enhance_function=Fusion_enhance,
    #     result_csv_path="tmp.csv"
    # )

    
    # 跑 method1
    # run_experiment(
    #     input_dir="/root/workspace/UIE/datasets/U45_dataset/upload/U45/U45",
    #     output_dir="/root/workspace/UIE/expri/U45/OURS",
    #     method_name="OURS",
    #     enhance_function=Adaptive_enhance,
    #     result_csv_path="evaluation_metrics.csv"
    # )

    # run_experiment(
    #     input_dir="/root/workspace/UIE/datasets/U45_dataset/upload/U45/U45",
    #     output_dir="/root/workspace/UIE/expri/U45/FUSION2",
    #     method_name="FUSION2",
    #     enhance_function=Fusion_enhance,
    #     result_csv_path="tmp.csv"
    #     # result_csv_path="evaluation_metrics.csv"
    # )