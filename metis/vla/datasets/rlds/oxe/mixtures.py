"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

# fmt: off
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === Human Ego Data for Dex Uni Token ===
    "egoatlas": [ 
        ("h2o", 1.0),
        ("ph2d", 1.0),
        ("oakink", 1.0),
        ("arctic", 1.0),
        ("egodex_part1", 0.038),
        ("egodex_part2", 0.039),
        ("egodex_part3", 0.042),
        ("egodex_part4", 0.041),
        ("egodex_part5", 0.040),
        ("fourier_all", 0.14),
        # ("hrdverse", 0.7)
        # ("fourier", 0.13),
        # ("fourier_pc", 0.18),


        # ("h2o_subject1_h1", 1.0),
        # ("h2o_subject1_h2", 1.0),
        # ("h2o_subject1_k1", 1.0),
        # ("h2o_subject1_k2", 1.0),
        # ("h2o_subject1_o1", 1.0),
        # ("h2o_subject1_o2", 1.0),
        # ("h2o_subject2_h1", 1.0),
        # ("h2o_subject2_h2", 1.0),
        # ("h2o_subject2_k1", 1.0),
        # ("h2o_subject2_k2", 1.0),
        # ("h2o_subject2_o1", 1.0),
        # ("h2o_subject2_o2", 1.0),
        # ("h2o_subject3_h1", 1.0),
        # ("h2o_subject3_h2", 1.0),
        # ("h2o_subject3_k1", 1.0),
        # ("h2o_subject3_k2", 1.0),
        # ("h2o_subject3_o1", 1.0),
        # ("h2o_subject3_o2", 1.0),
        
        # ("ph2d_grasp", 1.0),
        # ("ph2d_grasp_chocolate", 1.0),
        # ("ph2d_grasp_coke_random", 1.0),
        # ("ph2d_grasp_lars", 1.0),
        # ("ph2d_grasp_water", 1.0),
        # ("ph2d_grasp_pepsi", 1.0),
        # ("ph2d_grasp_zedbox", 1.0),
        # ("ph2d_grasping_mtn_bottle", 1.0),
        # ("ph2d_grasping_three_items", 1.0),
        # ("ph2d_pick", 1.0),
        # ("ph2d_pick_blackcube", 1.0),
        # ("ph2d_pick_brownbox", 1.0),
        # ("ph2d_pick_color_pad_left", 1.0),
        # ("ph2d_pick_dynamixel", 1.0),
        # ("ph2d_pick_on_color_pad_left", 1.0),
        # ("ph2d_pick_on_color_pad_right", 1.0),
        # ("ph2d_pick_on_color_pad_right_far", 1.0),
        # ("ph2d_pick_on_color_pad_right_far_far", 1.0),
        # ("ph2d_pick_orange", 1.0),
        # ("ph2d_picking_three_cat_straw", 1.0),
        # ("ph2d_pour_kura", 1.0),
        # ("ph2d_pour", 1.0),
        # ("ph2d_pour_mtn", 1.0),
        # ("ph2d_pour_party_cup", 1.0),
        # ("ph2d_pour_random", 1.0),
        # ("ph2d_pour_tea", 1.0),
        # ("ph2d_pour_water", 1.0),
        # ("ph2d_pouring_costco_water", 1.0),
        
        # ("egodex_part1_add_remove_lid", 0.05),
        # ("egodex_part1_arrange_topple_dominoes", 0.1),
        # ("egodex_part1_assemble_disassemble_legos", 0.03),
        # ("egodex_part1_assemble_disassemble_soft_legos", 0.08),
        # ("egodex_part1_assemble_disassemble_structures", 0.1),
        # ("egodex_part1_assemble_disassemble_tiles", 0.1),
        # ("egodex_part1_assemble_jenga", 0.1),
        # ("egodex_part1_boil_serve_egg", 0.1),
        # ("egodex_part1_braid_unbraid", 0.05),
        # ("egodex_part1_build_unstack_lego", 0.05),
        # ("egodex_part1_charge_uncharge_airpods", 0.01),
        # ("egodex_part1_charge_uncharge_device", 0.01),
        # ("egodex_part1_clean_cups", 0.01),
        # ("egodex_part1_clean_surface", 0.01),
        # ("egodex_part1_clean_tableware", 0.02),
        # ("egodex_part1_clip_unclip_papers", 0.1),
        # ("egodex_part1_color", 0.02),
        # ("egodex_part1_crumple_flatten_paper", 0.1),
        # ("egodex_part1_deal_gather_cards", 0.05),
        # ("egodex_part1_declutter_desk", 0.05),
        # ("egodex_part1_dry_hands", 0.02),
        # ("egodex_part1_fidget_magnetic_spinner_rings", 0.05),
        # ("egodex_part1_flip_coin", 0.3),
        # ("egodex_part1_flip_pages", 0.03),
        # ("egodex_part1_fry_bread", 1.0),
        # ("egodex_part1_fry_egg", 1.0),
        
        # ("egodex_part2_assemble_disassemble_furniture_bench_chair", 0.01),
        # ("egodex_part2_assemble_disassemble_furniture_bench_desk", 0.01),
        # ("egodex_part2_assemble_disassemble_furniture_bench_drawer", 0.01),
        # ("egodex_part2_assemble_disassemble_furniture_bench_lamp", 0.02),
        # ("egodex_part2_assemble_disassemble_furniture_bench_square_table", 0.01),
        # ("egodex_part2_assemble_disassemble_furniture_bench_stool", 0.01),
        # ("egodex_part2_basic_fold", 0.01),
        # ("egodex_part2_basic_pick_place", 0.03),
        # ("egodex_part2_fold_stack_unstack_unfold_cloths", 0.03),
        # ("egodex_part2_fold_unfold_paper_basic", 0.03),
        # ("egodex_part2_fold_unfold_paper_origami", 0.05),
        # ("egodex_part2_insert_remove_furniture_bench_cabinet", 0.01),
        # ("egodex_part2_insert_remove_furniture_bench_round_table", 0.01),
        
        # ("egodex_part3_gather_roll_dice", 0.03),
        # ("egodex_part3_insert_dump_blocks", 0.05),
        # ("egodex_part3_insert_remove_airpods", 0.02),
        # ("egodex_part3_insert_remove_bagging", 0.05),
        # ("egodex_part3_insert_remove_bookshelf", 0.02),
        # ("egodex_part3_insert_remove_cups_from_rack", 0.03),
        # ("egodex_part3_insert_remove_drawer", 0.03),
        # ("egodex_part3_insert_remove_plug_socket", 0.02),
        # ("egodex_part3_insert_remove_shirt_in_tube", 0.03),
        # ("egodex_part3_insert_remove_tennis_ball", 0.05),
        # ("egodex_part3_insert_remove_usb", 0.01),
        # ("egodex_part3_insert_remove_utensils", 0.02),
        # ("egodex_part3_knead_slime", 0.03),
        # ("egodex_part3_load_dispense_ice", 0.1),
        # ("egodex_part3_lock_unlock_key", 0.03),
        # ("egodex_part3_make_sandwich", 0.03),
        # ("egodex_part3_measure_objects", 0.1),
        # ("egodex_part3_open_close_insert_remove_box", 0.05),
        # ("egodex_part3_open_close_insert_remove_case", 0.03),
        # ("egodex_part3_open_close_insert_remove_tupperware", 0.02),
        # ("egodex_part3_paint_clean_brush", 0.03),
        # ("egodex_part3_peel_place_sticker", 0.05),
        # ("egodex_part3_pick_up_and_put_down_case_or_bag", 0.06),
        # ("egodex_part3_play_piano", 0.02),
        
        # ("egodex_part4_pick_place_food", 0.02),
        # ("egodex_part4_play_mancala", 0.02),
        # ("egodex_part4_play_reset_connect_four", 0.03),
        # ("egodex_part4_point_and_click_remote", 0.03),
        # ("egodex_part4_pour", 0.03),
        # ("egodex_part4_push_pop_toy", 0.02),
        # ("egodex_part4_put_away_set_up_board_game", 0.01),
        # ("egodex_part4_put_in_take_out_glasses", 0.02),
        # ("egodex_part4_put_toothpaste_on_toothbrush", 0.03),
        # ("egodex_part4_rake_smooth_zen_garden", 0.03),
        # ("egodex_part4_roll_ball", 0.05),
        # ("egodex_part4_scoop_dump_ice", 0.01),
        # ("egodex_part4_screw_unscrew_allen_fixture", 0.05),
        # ("egodex_part4_screw_unscrew_bottle_cap", 0.03),
        # ("egodex_part4_screw_unscrew_fingers_fixture", 0.02),
        # ("egodex_part4_set_up_clean_up_chessboard", 0.3),
        # ("egodex_part4_sleeve_unsleeve_cards", 0.2),
        # ("egodex_part4_slot_batteries", 0.05),
        # ("egodex_part4_sort_beads", 0.05),
        # ("egodex_part4_staple_paper", 0.1),
        # ("egodex_part4_stock_unstock_fridge", 0.05),
        # ("egodex_part4_sweep_dustpan", 0.1),
        
        # ("egodex_part5_stack", 0.01),
        # ("egodex_part5_stack_remove_jenga", 0.03),
        # ("egodex_part5_stack_unstack_bowls", 0.02),
        # ("egodex_part5_stack_unstack_cups", 0.05),
        # ("egodex_part5_stack_unstack_plates", 0.01),
        # ("egodex_part5_stack_unstack_tupperware", 0.05),
        # ("egodex_part5_thread_unthread_bead_necklace", 0.02),
        # ("egodex_part5_throw_and_catch_ball", 0.03),
        # ("egodex_part5_throw_collect_objects", 0.02),
        # ("egodex_part5_tie_and_untie_shoelace", 0.03),
        # ("egodex_part5_tie_untie_rubberband", 0.02),
        # ("egodex_part5_type_keyboard", 0.05),
        # ("egodex_part5_use_chopsticks", 0.1),
        # ("egodex_part5_use_rubiks_cube", 0.01),
        # ("egodex_part5_vertical_pick_place", 0.01),
        # ("egodex_part5_wash_fruit", 1.0),
        # ("egodex_part5_wash_kitchen_dishes", 1.0),
        # ("egodex_part5_wash_put_away_dishes", 1.0),
        # ("egodex_part5_wipe_kitchen_surfaces", 0.2),
        # ("egodex_part5_wipe_screen", 0.1),
        # ("egodex_part5_wrap", 0.05),
        # ("egodex_part5_wrap_unwrap_food", 0.03),
        # ("egodex_part5_write", 0.03),
        # ("egodex_part5_zip_unzip_bag", 0.02),
        # ("egodex_part5_zip_unzip_case", 0.03),

        # ("add_remove_lid", 1.0),
        # ("arrange_topple_dominoes", 1.0),
        # ("assemble_disassemble_legos", 1.0),
        # ("assemble_disassemble_soft_legos", 1.0),
        # ("assemble_disassemble_structures", 1.0),
        # ("assemble_disassemble_tiles", 1.0),
        # ("assemble_jenga", 1.0),
        # ("boil_serve_egg", 1.0),
        # ("braid_unbraid", 1.0),
        # ("build_unstack_lego", 1.0),
        # ("charge_uncharge_airpods", 1.0),
        # ("charge_uncharge_device", 1.0),
        # ("clean_cups", 1.0),
        # ("clean_surface", 1.0),
        # ("clean_tableware", 1.0),
        # ("clip_unclip_papers", 1.0),
        # ("color", 1.0),
        # ("crumple_flatten_paper", 1.0),
        # ("deal_gather_cards", 1.0),
        # ("declutter_desk", 1.0),
        # ("dry_hands", 1.0),
        # ("fidget_magnetic_spinner_rings", 1.0),
        # ("flip_coin", 1.0),
        # ("flip_pages", 1.0),
        # ("fry_bread", 1.0),
        # ("fry_egg", 1.0),
    ],
    # === Real Robot Data ===
    "robot": [ 
        # ("pick_up_the_cup_and_pour_it_into_the_container", 1.0),
        # ("pick_up_the_cup_and_place_it_in_the_basket", 1.0),
        # ("pick_up_the_croissant_on_the_table_and_put_it_in_the_container", 1.0),
        # ("pick_up_the_croissant_from_the_table_and_place_it_in_the_container", 1.0),
        # ("pick_up_the_mouse_from_the_table_and_place_it_in_the_container", 1.0),
        # ("pick_up_the_mouse_and_put_it_in_the_white_basket", 1.0),
        # ("put_the_yellow_tape_measure_into_the_white_plastic_container", 1.0),
        # ("put_the_yellow_tape_measure_in_the_white_basket", 1.0),
        # ("put_the_yellow_tape_measure_in_the_black_basket", 1.0),
        # ("put_the_purple_eggplant_in_the_white_pan", 1.0),
        # ("put_the_purple_eggplant_in_the_white_basket", 1.0),
        # ("put_the_pink_cup_in_the_pan", 1.0),
        # ("put_the_pink_cup_in_the_basket", 1.0),
        # ("put_the_lettuce_in_the_container", 1.0),
        # ("put_the_cup_in_the_container", 1.0),
        ("grab_the_apple_and_put_it_into_the_basket", 1.0),
    ],
    
    "oxe": [ 
        ("fractal20220817_data", 1.0),
    ],
    # === Bridge V2 Dataset ===
    "bridge": [
        ("bridge_oxe", 1.0),                                      # Version of Bridge V2 in Open-X GCP Bucket
        # ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],

    "droid": [
        ("droid", 1.0),
    ],
    
    # === Human-data Only ===
    # "Ego4D": [ 
    #     ("ego4d_split_1", 1.0),
    #     ("ego4d_split_2", 1.0),
    #     ("ego4d_split_3", 1.0),
    #     ("ego4d_split_4", 1.0),
    # ],


    "roboset": [
        ("roboset", 1.0),
    ],

    "stanford_robocook_converted_externally_to_rlds": [
        ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ],

    # === [Moderate-Scale] Bridge++ Mixtures ===
    "bridge_rt_1": [
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
    ],

    "rt_1": [
        ("fractal20220817_data", 1.0),
    ],

    # === T-DROID Dataset ===
    "tdroid_carrot_in_bowl": [
        ("tdroid_carrot_in_bowl", 1.0),
    ],
    "tdroid_pour_corn_in_pot": [
        ("tdroid_pour_corn_in_pot", 1.0),
    ],
    "tdroid_flip_pot_upright": [
        ("tdroid_flip_pot_upright", 1.0),
    ],
    "tdroid_move_object_onto_plate": [
        ("tdroid_move_object_onto_plate", 1.0),
    ],
    "tdroid_knock_object_over": [
        ("tdroid_knock_object_over", 1.0),
    ],
    "tdroid_cover_object_with_towel": [
        ("tdroid_cover_object_with_towel", 1.0),
    ],

    # === DROID Finetuning Datasets ===
    "droid_wipe": [
        ("droid_wipe", 1.0),
    ],

    # === LIBERO Datasets (Modified Versions) ===
    "libero_spatial_no_noops": [
        ("libero_spatial_no_noops", 1.0),
    ],
    "libero_object_no_noops": [
        ("libero_object_no_noops", 1.0),
    ],
    "libero_goal_no_noops": [
        ("libero_goal_no_noops", 1.0),
    ],
    "libero_10_no_noops": [
        ("libero_10_no_noops", 1.0),
    ],
    "libero_10_no_noops_mini": [
        ("libero_10_no_noops_mini", 1.0),
    ],
    "libero_goal_no_noops_mini": [
        ("libero_goal_no_noops_mini", 1.0),
    ],
    "libero_goal_no_noops_half": [
        ("libero_goal_no_noops_half", 1.0),
    ],
    "libero_10_no_noops_half": [
        ("libero_10_no_noops_half", 1.0),
    ],
    "libero_goal_no_noops_quad": [
        ("libero_goal_no_noops_quad", 1.0),
    ],
    "libero_10_no_noops_quad": [
        ("libero_10_no_noops_quad", 1.0),
    ],
    "libero_combined": [
        ("libero_combined", 1.0),
    ],
}
# fmt: on
