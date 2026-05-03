import os
import json
import re

class ManageJson:
    @staticmethod
    def collects_json_files_for_ui(path_my_dataset_ui, path_rico_json_semantic, path_output_json_semantic):
        list_topics_folders = [d for d in os.listdir(path_my_dataset_ui) if os.path.isdir(os.path.join(path_my_dataset_ui, d))]
        list_sem_json_from_rico = [f for f in os.listdir(path_rico_json_semantic) if os.path.isfile(os.path.join(path_rico_json_semantic, f))]
        error = 0

        for i in range(1, len(list_topics_folders)):
            topic_folder = os.path.join(path_my_dataset_ui, list_topics_folders[i])
            all_files_ui = [f for f in os.listdir(topic_folder) if os.path.isfile(os.path.join(topic_folder, f))]
            path_one_topic = os.path.join(path_output_json_semantic, list_topics_folders[i])
            os.makedirs(path_one_topic, exist_ok=True)

            for file_ui in all_files_ui:
                name_ui = os.path.splitext(file_ui)[0]
                try:
                    path_s = next((os.path.join(path_rico_json_semantic, f) for f in list_sem_json_from_rico if f == f"{name_ui}.json"), None)
                    if path_s:
                        path_file_des = os.path.join(path_one_topic, f"{name_ui}.json")
                        os.rename(path_s, path_file_des)
                except Exception as h:
                    with open(os.path.join(path_rico_json_semantic, "error.txt"), 'a') as error_file:
                        error_file.write(f"{name_ui}\n")
                    error += 1

        print(f"Errors: {error}")

    @staticmethod
    def delete_ui_not_has_json(path_my_dataset_ui):
        with open(r"D:\erorr.txt", 'r') as file:
            ui_list = file.readlines()
        list_ui = [line.strip() for line in ui_list]
        list_topics_folders = [d for d in os.listdir(path_my_dataset_ui) if os.path.isdir(os.path.join(path_my_dataset_ui, d))]

        total_ui = 0
        for topic_folder in list_topics_folders:
            topic_path = os.path.join(path_my_dataset_ui, topic_folder)
            all_files = [f for f in os.listdir(topic_path) if os.path.isfile(os.path.join(topic_path, f))]
            for file_ui in all_files:
                name_ui = os.path.splitext(file_ui)[0]
                if name_ui in list_ui:
                    os.remove(os.path.join(topic_path, file_ui))
                    total_ui += 1

        print(f"Total UI deleted: {total_ui}")

    @staticmethod
    def get_count_of_files_in_folder(path_folder):
        list_topics_folders = [d for d in os.listdir(path_folder) if os.path.isdir(os.path.join(path_folder, d))]
        line = ""
        total_ui = 0

        for i, topic_folder in enumerate(list_topics_folders):
            topic_path = os.path.join(path_folder, topic_folder)
            all_files = [f for f in os.listdir(topic_path) if os.path.isfile(os.path.join(topic_path, f))]
            line += f"{i + 1}-    {topic_folder}\t{len(all_files)}\n"
            total_ui += len(all_files)

        line += "\n-------------------------------------\n"
        line += f"Total UI = {total_ui}"
        with open(os.path.join(path_folder, "Details.txt"), 'w') as file:
            file.write(line)

    @staticmethod
    def get_all_children(parent):
        all_children = []
        children = parent.get("children", [])
        for child in children:
            all_children.append(child)
            nested_children = ManageJson.get_all_children(child)
            all_children.extend(nested_children)
        return all_children

    @staticmethod
    def get_all_children_full(parent):
        all_children = []
        children = parent.get("activity", {}).get("root", {}).get("children", [])
        for child in children:
            if child:
                all_children.append(child)
                nested_children = ManageJson.get_all_children(child)
                all_children.extend(nested_children)
        return all_children

    @staticmethod
    def get_activity_name(json_file_path):
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
                activity_name = data.get("activity_name", "")
                if activity_name:
                    parts = activity_name.split('.')
                    if parts:
                        return parts[-1]
            return ""
        except FileNotFoundError:
            return ""

    @staticmethod
    def get_text_from_child(child):
        words = ""
        txt1 = child.get("text", "")
        txt2 = child.get("textButtonClass", "")
        txt3 = child.get("iconClass", "")
        txt4 = child.get("resource-id", "")
        txt5 = child.get("class", "")
        txt6 = child.get("text-hint", "")

        if txt1:
            words += ManageJson.clean_text(txt1) + " "
        if txt2:
            words += ManageJson.clean_text(txt2) + " "
        if txt3:
            words += ManageJson.clean_text(txt3) + " "
        if txt4:
            fff = txt4.split('/')[1] if '/' in txt4 else txt4
            if '_' in fff:
                firstw, secw = fff.split('_')
                words += ManageJson.clean_text(f"{fff} {firstw} {secw}") + " "
            else:
                words += ManageJson.clean_text(fff) + " "
        if txt5:
            words += ManageJson.clean_text(txt5) + " "
        if txt6:
            words += ManageJson.clean_text(txt6) + " "
        return words

    @staticmethod
    def clean_text(input_text):
        pattern = r"[^A-Za-z0-9@#$&*\/]"
        cleaned_text = re.sub(pattern, "", input_text)
        return cleaned_text