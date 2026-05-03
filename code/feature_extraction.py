import os
import json
import random
from ManageJson import ManageJson
from ManageKeywords import ManageKeywords


class Manage_MASC:
    screen_count = 0
    list_clickable = ["Checkbox", "Icon", "Button Bar", "Image", "Radio Button", "Text Button", "On / Off Switch", "Number Stepper",
                      "Drawer", "Bottom Navigation", "Date Picker", "Modal", "Video", "List Item", "Advertisement", "Web View"]
    list_input = ["Input"]
    list_swipe_vertical = ["List Item"]
    list_swipe_horizontal = ["List Item", "Multi - Tab", "Bottom Navigation", "Slider", "Pager Indicator", "On / Off Switch"]

    @staticmethod
    def search_list_strings_in_text(strings_to_search, text):
        found_strings = ""
        for search_string in strings_to_search:
            if search_string.lower() in text.lower():
                if search_string not in found_strings:
                    found_strings += search_string + " "
        return found_strings.strip()

    @staticmethod
    def get_all_features(list_vector, list_features, list_labels, path_of_json, path_of_full_json, is_selected_features=True):
        if is_selected_features:
            list_vector.append("Screen Id,Number of Clickable Elements - Middle,Number of General Elements,Number of Clickable Elements - Bottom,"
                               "Number of Clickable Elements - Top,Number of vertical swipeable Elements - Middle,"
                               "is the activity contain a navigation drawer,Number of Text Fields Elements - Middle,Label")
            list_features.append("Number of Clickable Elements - Middle,Number of General Elements,Number of Clickable Elements - Bottom,"
                                  "Number of Clickable Elements - Top,Number of vertical swipeable Elements - Middle,"
                                  "is the activity contain a navigation drawer,Number of Text Fields Elements - Middle,keywords")
            list_labels.append("Screen Id,class")
        else:
            list_vector.append("Screen Id,Number of Clickable Elements - Middle,Number of General Elements,Number of Long - Clickable Elements,"
                               "Number of Clickable Elements - Bottom,Number of Clickable Elements - Top,Number of vertical swipeable Elements - Middle,"
                               "is the activity contain a navigation drawer,Number of Text Fields Elements - Middle,Number of horizontal swipeable Elements - Middle,"
                               "Number of Text Fields Elements - Bottom,Number of horizontal swipeable Elements - Top,Number of horizontal swipeable Elements - Bottom,"
                               "Number of vertical swipeableElements - Bottom,Number of vertical swipeableElements - Top,Number of Text Fields Elements - Top,Words,Label")
            list_features.append("Number of Clickable Elements - Middle,Number of General Elements,"
                                  "Number of Clickable Elements - Bottom,Number of Clickable Elements - Top,"
                                  "Number of vertical swipeable Elements - Middle,is the activity contain a navigation drawer,"
                                  "Number of Text Fields Elements - Middle,Number of horizontal swipeable Elements - Middle,"
                                  "Number of Text Fields Elements - Bottom,Number of Text Fields Elements - Top,keywords")
            list_labels.append("Screen Id,class")

        folders = os.listdir(path_of_json)
        folders_full = os.listdir(path_of_full_json)

        # FIX 1: Filter out files like Details.txt - only keep folders
        folders = [f for f in folders if os.path.isdir(os.path.join(path_of_json, f))]
        folders_full = [f for f in folders_full if os.path.isdir(os.path.join(path_of_full_json, f))]

        top_height_percentage = 0.15
        middle_height_percentage = 0.7
        bottom_height_percentage = 0.15

        for j, category_semantic in enumerate(folders):
            category_semantic_path = os.path.join(path_of_json, category_semantic)
            all_files_semantic = os.listdir(category_semantic_path)
            class_name = category_semantic
            category_full_path = os.path.join(path_of_full_json, folders_full[j])
            all_files_full = os.listdir(category_full_path)

            for i, file_semantic in enumerate(all_files_semantic):
                # FIX 2: Skip if index out of range
                if i >= len(all_files_full):
                    continue

                activity_name = ManageJson.get_activity_name(os.path.join(category_full_path, all_files_full[i]))
                Manage_MASC.screen_count += 1
                screen_id = os.path.splitext(file_semantic)[0]
                json_file_path = os.path.join(category_semantic_path, file_semantic)

                try:
                    # FIX 3: utf-8 encoding to handle special characters
                    with open(json_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        data = json.load(file)
                        bounds = data.get("bounds", [])

                        if bounds:
                            activity_height = bounds[3] - bounds[1]
                            top_height = int(activity_height * top_height_percentage)
                            middle_height = int(activity_height * middle_height_percentage)
                            bottom_height = int(activity_height * bottom_height_percentage)

                            all_children = ManageJson.get_all_children(data)

                            ct_count = 0
                            cm_count = 0
                            cb_count = 0
                            svt_count = 0
                            svm_count = 0
                            svb_count = 0
                            sht_count = 0
                            shm_count = 0
                            shb_count = 0
                            g_count = 0
                            lc_count = 0
                            is_drawer = 0
                            in_t_count = 0
                            in_m_count = 0
                            in_b_count = 0

                            if all_children:
                                keywords = ""
                                # FIX 3: Same encoding fix for full json file
                                with open(os.path.join(category_full_path, all_files_full[i]), 'r', encoding='utf-8', errors='ignore') as full_file:
                                    data_full = json.load(full_file)
                                    all_children_full = ManageJson.get_all_children_full(data_full)
                                    for child in all_children_full:
                                        keywords += ManageJson.get_text_from_child(child)

                                for child in all_children:
                                    top_bound = child["bounds"][1]
                                    # FIX 4: Handle clickable as bool or string
                                    clickable_val = child.get("clickable", "false")
                                    clickable_str = str(clickable_val).lower()

                                    if Manage_MASC.list_clickable.__contains__(child["componentLabel"]) and clickable_str == "true":
                                        if top_bound < top_height:
                                            ct_count += 1
                                        elif top_bound < top_height + middle_height:
                                            cm_count += 1
                                        else:
                                            cb_count += 1
                                        if child.get("iconClass", "").lower() == "menu":
                                            is_drawer += 1
                                        g_count += 1
                                    elif Manage_MASC.list_input.__contains__(child["componentLabel"]):
                                        if top_bound < top_height:
                                            in_t_count += 1
                                        elif top_bound < top_height + middle_height:
                                            in_m_count += 1
                                        else:
                                            in_b_count += 1
                                        g_count += 1
                                    elif Manage_MASC.list_swipe_vertical.__contains__(child["componentLabel"]):
                                        if top_bound < top_height:
                                            svt_count += 1
                                        elif top_bound < top_height + middle_height:
                                            svm_count += 1
                                        else:
                                            svb_count += 1
                                        g_count += 1
                                    elif Manage_MASC.list_swipe_horizontal.__contains__(child["componentLabel"]):
                                        if top_bound < top_height:
                                            sht_count += 1
                                        elif top_bound < top_height + middle_height:
                                            shm_count += 1
                                        else:
                                            shb_count += 1
                                        g_count += 1

                                # FIX 5: Handle missing keyword category gracefully
                                list_keywords = ManageKeywords.keywords_dictionary.get(category_semantic, [])
                                keywords = Manage_MASC.search_list_strings_in_text(list_keywords, keywords.lower() + activity_name.lower())

                                if is_selected_features:
                                    list_vector.append(f"{screen_id},{cm_count},{g_count},{cb_count},{ct_count},{svm_count},{is_drawer},{in_m_count},{j + 1}")
                                    list_features.append(f"{cm_count},{g_count},{cb_count},{ct_count},{svm_count},{is_drawer},{in_m_count}")
                                    list_labels.append(f"{screen_id},{j + 1}")
                                else:
                                    list_vector.append(f"{screen_id},{cm_count},{g_count},{lc_count},{cb_count},{ct_count},{svm_count},{is_drawer},{in_m_count},{shm_count},{in_b_count},{sht_count},{shb_count},{svb_count},{svt_count},{in_t_count},{keywords},{class_name}")
                                    list_features.append(f"{cm_count},{g_count},{cb_count},{ct_count},{svm_count},{is_drawer},{in_m_count},{shm_count},{in_b_count},{in_t_count},{keywords}")
                                    list_labels.append(f"{screen_id},{class_name}")
                        else:
                            print("No 'bounds' array found in the JSON.")
                except Exception as ex:
                    print(f"An error occurred: {ex}")

    @staticmethod
    def get_number_apps_of_ui():
        try:
            path = r"D:\Research\MyWork\NewTopic\Datasets\Dataset_Rico\ui_details.csv"
            path2 = r"D:\Research\MyWork\NewTopic\Datasets\Dataset_EnRico\Dataset_EnRico\design_topics.csv"

            with open(path, 'r') as file:
                lines = file.readlines()
            with open(path2, 'r') as file:
                lines2 = file.readlines()

            list_ui = [line.split(',')[0] for line in lines[1:]]
            list_names = [line.split(',')[1] for line in lines[1:]]
            list_ui2 = [line.split(',')[0] for line in lines2[1:]]

            list_distinct_names = list(set(list_names[i] for i, ui in enumerate(list_ui) if ui in list_ui2))

            print(len(list_distinct_names))
        except Exception as ex:
            print(f"An error occurred: {ex}")

    def run_masc(self):
        path_json_test = os.path.join("MASC_Json", "MASC_Json", "Semantic")
        path_json_test_full = os.path.join("MASC_Json", "MASC_Json", "Full")
        path_json_output = os.path.join("data", "processed")

        list_vector_selected = []
        list_features_selected = []
        list_labels_selected = []

        list_vector_full = []
        list_features_full = []
        list_labels_full = []

        Manage_MASC.get_all_features(
            list_vector_selected, list_features_selected, list_labels_selected,
            path_json_test, path_json_test_full, is_selected_features=True
        )

        Manage_MASC.get_all_features(
            list_vector_full, list_features_full, list_labels_full,
            path_json_test, path_json_test_full, is_selected_features=False
        )

        os.makedirs(path_json_output, exist_ok=True)

        with open(os.path.join(path_json_output, "Allvectors.csv"), 'w', newline='') as f:
            f.writelines(line + '\n' for line in list_vector_full)

        with open(os.path.join(path_json_output, "dataset.csv"), 'w', newline='') as f:
            f.writelines(line + '\n' for line in list_features_full)

        with open(os.path.join(path_json_output, "Labels.csv"), 'w', newline='') as f:
            f.writelines(line + '\n' for line in list_labels_full)

        print(f"Done! Processed {Manage_MASC.screen_count} screens.")
        print(f"Files saved to: {path_json_output}")


# Outside the class - no indentation
if __name__ == "__main__":
    obj = Manage_MASC()
    obj.run_masc()