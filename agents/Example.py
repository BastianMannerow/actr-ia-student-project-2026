"""Production-centred ACT-R model for the cooperative platform environment.

The file contains only ACT-R chunks, declarative associations, imaginal
workspaces and productions.  Continuous visual interpretation is delegated to
``ExampleAdapter``.  The adapter can only access the model through
``pyactrFunctionalityExtension`` and is represented as a separate transition
layer in the State Graph; it is not an ACT-R module.
"""

from __future__ import annotations

import pyactr as actr


class Example:
    """Cognitive command centre for two-agent platform cooperation."""

    uses_visual_module = True
    perfect_line_of_sight = True
    analysis_control_slots = ("state", "strategy")

    def __init__(self, environ):
        self.environ = environ
        self.actr_agent = actr.ACTRModel(
            environment=environ,
            motor_prepared=True,
            automatic_visual_search=False,
            subsymbolic=True,
            baselevel_learning=True,
            optimized_learning=True,
            latency_factor=0.06,
            retrieval_threshold=-12.0,
            instantaneous_noise=0.0,
            utility_noise=0.08,
            utility_learning=False,
        )
        self._define_chunk_types()
        self.initial_goal = actr.chunkstring(
            string="""
                isa controller
                state identify_request
                self_symbol unknown
                self_shape unknown
                target_id none
                strategy none
                cycle 0
                last_action none
                coordination_phase none
            """
        )

    def build_agent(self, _agent_list):
        model = self.actr_agent
        for buffer_name in (
            "imaginal_self",
            "imaginal_other",
            "imaginal_world",
            "imaginal_target",
            "imaginal_affordance",
            "imaginal_coordination",
            "imaginal_progress",
            "imaginal_recovery",
        ):
            model.set_goal(name=buffer_name, delay=0)

        model.goals["imaginal_self"].add(actr.makechunk(
            typename="self_model", identity="unknown", shape="unknown", x="0", y="0",
            previous_x="0", previous_y="0", dx="0", dy="0", grounded="no",
            support="none", transform="0", region="unknown", motion="stationary",
            confidence="0", revision="0"))
        model.goals["imaginal_other"].add(actr.makechunk(
            typename="other_model", identity="unknown", shape="unknown", x="0", y="0",
            previous_x="0", previous_y="0", dx="0", dy="0", grounded="no",
            support="none", transform="0", region="unknown", motion="stationary",
            inferred_intent="unknown", inferred_target="none", readiness="no",
            relation="unknown", blocking="none", self_blocking="no",
            partner_blocking="no", commitment="no", confidence="0", revision="0"))
        model.goals["imaginal_world"].add(actr.makechunk(
            typename="world_model", revision="0", remaining="0", upper_remaining="0",
            lower_remaining="0", region="unknown", priority_pending="no",
            safe_to_descend="no", irreversible_drop="no", fall_risk="no",
            cooperation_possible="no", agents_separated="no", reunion_required="no",
            room_conflict="no", status="unknown"))
        model.goals["imaginal_target"].add(actr.makechunk(
            typename="candidate_model", status="none", target_id="none", role="none",
            reachability="none", actor="none", strategy="idle_monitor",
            cooperation="no", priority="no", score="0", distance="999", dx="0", dy="0",
            staging_x="0", staging_y="0", reason="none", stagnation="0",
            reservation="none", revision="0"))
        model.goals["imaginal_affordance"].add(actr.makechunk(
            typename="affordance_model", transform_up="no",
            transform_down="no", move_left="no", move_right="no",
            prime_jump_left="no", prime_jump_right="no", fast_fall="no", wait="yes",
            replan="no", partner_ready="no", circle_on_rectangle="no", aligned="no",
            target_above="no", unsafe_descent="no", reason="initial", revision="0"))
        model.goals["imaginal_coordination"].add(actr.makechunk(
            typename="coordination_model", joint_target="none", phase="idle_monitor",
            self_role="observer", partner_role="observer", self_commitment="no",
            partner_commitment="no", route_conflict="no", blocker="none",
            yield_direction="level", partner_direction="level", separation="0",
            progress="initializing", timeout="0", association="perfect_los", revision="0"))
        model.goals["imaginal_progress"].add(actr.makechunk(
            typename="progress_model", target_id="none", current_x="0", current_y="0",
            previous_x="0", previous_y="0", current_distance="999",
            previous_distance="999", route_distance="999", previous_route_distance="999",
            best_route_distance="999", displacement="0", distance_gain="0",
            route_gain="0", no_progress_cycles="0", repeated_action="none",
            repeated_count="0", last_direction="level", direction_changes="0",
            approach_side="none", support="none", previous_support="none",
            failed_attempts="0", irrational_cycle="no", stuck="no",
            stuck_kind="none", cause="none", recovery_required="no", revision="0"))
        model.goals["imaginal_recovery"].add(actr.makechunk(
            typename="recovery_plan", stuck_kind="none", strategy="none", action="none",
            direction="none", association="none", target_id="none", approach_side="none",
            attempts="0", status="idle", revision="0"))

        self._seed_declarative_chunks(model)
        self._define_productions(model)
        return model

    @staticmethod
    def _define_chunk_types() -> None:
        chunk_types = {
            "controller": "state self_symbol self_shape target_id strategy cycle last_action coordination_phase",
            "identity_model": "identity shape status revision",
            "self_model": "identity shape x y previous_x previous_y dx dy grounded support transform region motion confidence revision",
            "other_model": "identity shape x y previous_x previous_y dx dy grounded support transform region motion inferred_intent inferred_target readiness relation blocking self_blocking partner_blocking commitment confidence revision",
            "world_model": "revision remaining upper_remaining lower_remaining region priority_pending safe_to_descend irreversible_drop fall_risk cooperation_possible agents_separated reunion_required room_conflict status",
            "target_model": "target_id role x y region required_order reachability actor strategy cooperation priority status score distance dx dy staging_x staging_y reason revision",
            "candidate_model": "status target_id role reachability actor strategy cooperation priority score distance dx dy staging_x staging_y reason stagnation reservation revision",
            "affordance_model": "transform_up transform_down move_left move_right prime_jump_left prime_jump_right fast_fall wait replan partner_ready circle_on_rectangle aligned target_above unsafe_descent reason revision",
            "coordination_model": "joint_target phase self_role partner_role self_commitment partner_commitment route_conflict blocker yield_direction partner_direction separation progress timeout association revision",
            "progress_model": "target_id current_x current_y previous_x previous_y current_distance previous_distance route_distance previous_route_distance best_route_distance displacement distance_gain route_gain no_progress_cycles repeated_action repeated_count last_direction direction_changes approach_side support previous_support failed_attempts irrational_cycle stuck stuck_kind cause recovery_required revision",
            "recovery_schema": "stuck_kind shape strategy action direction association priority",
            "recovery_plan": "stuck_kind strategy action direction association target_id approach_side attempts status revision",
            "coordination_schema": "phase recommended_strategy response association priority",
            "social_episode": "situation self_role partner_role action outcome association utility",
            "strategy_schema": "strategy objective self_role partner_role precondition coordination timing terminal",
            "environment_rule": "rule subject relation object condition consequence priority",
            "metric_parameter": "parameter value unit",
            "target_policy": "role circle_strategy rectangle_strategy priority_weight order_bias requires_cooperation",
            "platform_model": "platform_id kind left right top bottom width height region traversability revision",
        }
        for typename, slots in chunk_types.items():
            actr.chunktype(typename, slots)

    @staticmethod
    def _seed_declarative_chunks(model) -> None:
        rules = (
            ("gravity", "avatar", "falls_toward", "lower_surface", "airborne", "vertical_velocity_increases", "10"),
            ("circle_jump", "circle", "can_reach", "nearby_upper_surface", "grounded", "jump_then_steer", "20"),
            ("rectangle_transform", "rectangle", "changes_aspect_ratio", "height_and_width", "clearance_available", "w_taller_s_flatter", "20"),
            ("stack_cooperation", "circle_and_rectangle", "jointly_reach", "high_diamond", "circle_on_rectangle", "rectangle_raises_then_circle_jumps", "30"),
            ("priority_before_drop", "team", "must_collect", "upper_targets", "before_irreversible_descent", "delay_descent_until_upper_clear", "40"),
            ("perfect_los", "observer", "perceives", "all_relevant_entities", "line_of_sight_unoccluded", "maintain_situation_model", "15"),
            ("route_blocking", "team_member", "must_yield", "partner_corridor", "self_between_partner_and_target", "clear_route_before_waiting", "35"),
            ("joint_commitment", "team", "shares", "cooperative_target", "roles_are_complementary", "maintain_target_until_joint_phase_complete", "38"),
            ("reunion", "separated_agents", "must_restore", "shared_region", "cooperation_not_possible", "approach_partner_before_target_execution", "36"),
        )
        for values in rules:
            model.decmem.add(
                actr.makechunk(
                    typename="environment_rule",
                    rule=values[0],
                    subject=values[1],
                    relation=values[2],
                    object=values[3],
                    condition=values[4],
                    consequence=values[5],
                    priority=values[6],
                )
            )

        strategies = (
            ("direct_circle", "collect_reachable_target", "circle", "monitor", "target_not_blocked", "approach_prime_jump_collect", "target_collected"),
            ("direct_rectangle", "collect_ground_or_tunnel_target", "rectangle", "monitor", "target_at_rectangle_height", "adjust_aspect_ratio_then_approach", "target_collected"),
            ("cooperative_circle", "use_partner_as_mobile_platform", "circle", "rectangle", "partner_staged", "mount_wait_for_raise_then_jump", "cooperative_target_collected"),
            ("cooperative_rectangle", "provide_mobile_platform", "rectangle", "circle", "circle_ready_to_mount", "stage_wait_for_mount_then_raise", "cooperative_target_collected"),
            ("support_partner", "avoid_competing_for_partner_target", "support", "collector", "partner_has_better_actor_fit", "monitor_partner_and_preserve_route", "partner_target_resolved"),
            ("descend_after_priority", "cross_irreversible_drop", "any", "team", "upper_region_clear", "move_to_drop_and_fall_together", "lower_region_reached"),
            ("recover_replan", "recognize_unsolved_route", "any", "team", "upper_target_left_but_team_below", "stop_resetting_and_reclassify_available_targets", "new_route_or_explicit_blocked_state"),
            ("idle_monitor", "maintain_situational_awareness", "any", "team", "no_immediate_affordance", "wait_observe_reclassify", "new_affordance"),
            ("yield_route", "clear_partner_corridor", "support", "collector", "self_blocks_partner", "move_out_of_corridor_then_reassess", "partner_route_clear"),
            ("reunite_team", "restore_shared_operating_region", "separated", "partner", "agents_on_different_tiers_or_rooms", "approach_visible_partner_or_shared_exit", "team_reunited"),
        )
        for values in strategies:
            model.decmem.add(
                actr.makechunk(
                    typename="strategy_schema",
                    strategy=values[0],
                    objective=values[1],
                    self_role=values[2],
                    partner_role=values[3],
                    precondition=values[4],
                    coordination=values[5],
                    timing=values[5],
                    terminal=values[6],
                )
            )

        parameters = (
            ("visual_position_scale", "4.0", "quantized_per_world_unit"),
            ("position_tolerance", "1.6", "world_units"),
            ("stage_tolerance", "2.4", "world_units"),
            ("takeoff_tolerance", "0.35", "world_units"),
            ("jump_vertical_envelope", "10.5", "world_units"),
            ("cooperation_mount_window", "5.8", "world_units"),
            ("cooperation_center_tolerance", "1.15", "world_units"),
            ("cooperation_jump_window", "11.5", "world_units"),
            ("rectangle_min_height", "2.6", "world_units"),
            ("rectangle_max_height", "44.0", "world_units"),
            ("rectangle_flat_threshold", "0.15", "ratio"),
            ("rectangle_tall_threshold", "0.82", "ratio"),
            ("fall_risk_margin", "3.0", "world_units"),
            ("ground_contact_tolerance", "0.85", "world_units"),
            ("ledge_takeoff_offset", "4.8", "world_units"),
            ("jump_priming_clearance", "1.4", "world_units"),
            ("recovery_runway_increment", "0.9", "world_units_per_attempt"),
            ("route_progress_tolerance", "0.18", "world_units"),
            ("failed_jump_limit", "2", "attempts"),
            ("oscillation_limit", "3", "direction_changes"),
            ("cooperation_stage_offset", "5.0", "world_units"),
            ("blocking_distance", "10.0", "world_units"),
            ("corridor_vertical_tolerance", "4.0", "world_units"),
            ("yield_clearance", "8.0", "world_units"),
            ("reunion_distance", "12.0", "world_units"),
        )
        for parameter, value, unit in parameters:
            model.decmem.add(
                actr.makechunk(
                    typename="metric_parameter",
                    parameter=parameter,
                    value=value,
                    unit=unit,
                )
            )

        policies = (
            ("solo_circle", "direct_circle", "support_partner", "70", "0", "no"),
            ("cooperative_stack", "cooperative_circle", "cooperative_rectangle", "85", "0", "yes"),
            ("solo_flat_rectangle", "support_partner", "direct_rectangle", "78", "0", "no"),
            ("priority_before_drop", "direct_circle", "support_partner", "130", "1", "no"),
            ("solo_lower", "direct_circle", "support_partner", "68", "2", "no"),
            ("solo_ground", "direct_circle", "direct_rectangle", "64", "2", "no"),
        )
        for values in policies:
            model.decmem.add(
                actr.makechunk(
                    typename="target_policy",
                    role=values[0],
                    circle_strategy=values[1],
                    rectangle_strategy=values[2],
                    priority_weight=values[3],
                    order_bias=values[4],
                    requires_cooperation=values[5],
                )
            )

        coordination_associations = (
            ("direct_circle", "direct_circle", "execute_candidate", "shape_target_fit", "20"),
            ("direct_rectangle", "direct_rectangle", "execute_candidate", "shape_target_fit", "20"),
            ("cooperative_circle", "cooperative_circle", "maintain_rider_commitment", "joint_target_binding", "35"),
            ("cooperative_rectangle", "cooperative_rectangle", "maintain_provider_commitment", "joint_target_binding", "35"),
            ("support_partner", "support_partner", "monitor_without_obstruction", "complementary_role", "18"),
            ("yield_route", "yield_route", "clear_partner_corridor", "blocking_requires_yield", "45"),
            ("reunite_team", "reunite_team", "restore_shared_region", "separation_precedes_cooperation", "42"),
            ("descend_after_priority", "descend_after_priority", "cross_drop_together", "ordered_descent", "30"),
            ("recover_replan", "recover_replan", "reclassify_without_reset", "trapped_team", "50"),
            ("idle_monitor", "idle_monitor", "observe_and_reclassify", "no_actionable_target", "10"),
        )
        for phase, recommended, response, association, priority in coordination_associations:
            model.decmem.add(
                actr.makechunk(
                    typename="coordination_schema",
                    phase=phase,
                    recommended_strategy=recommended,
                    response=response,
                    association=association,
                    priority=priority,
                )
            )

        social_episodes = (
            ("partner_blocks_route", "collector", "support", "yield_or_jump_over", "route_progress", "blocking_requires_yield", "0.9"),
            ("self_blocks_partner", "support", "collector", "clear_corridor", "partner_progress", "blocking_requires_yield", "1.0"),
            ("circle_mounted", "provider", "rider", "raise_only_when_centered", "stable_lift", "joint_target_binding", "1.0"),
            ("rectangle_tall", "rider", "provider", "timed_jump", "target_access", "joint_target_binding", "1.0"),
            ("agents_separated", "separated", "partner", "reunite", "cooperation_possible", "separation_precedes_cooperation", "0.95"),
        )
        for values in social_episodes:
            model.decmem.add(
                actr.makechunk(
                    typename="social_episode",
                    situation=values[0],
                    self_role=values[1],
                    partner_role=values[2],
                    action=values[3],
                    outcome=values[4],
                    association=values[5],
                    utility=values[6],
                )
            )

        recovery_schemas = (
            ("immobile", "circle", "escape_jump", "jump", "dynamic", "stagnation_requires_escape", "45"),
            ("immobile", "rectangle", "reverse_and_resize", "reverse", "dynamic", "stagnation_requires_escape", "45"),
            ("oscillating", "circle", "extend_runway", "reverse", "subgoal", "oscillation_requires_new_runway", "48"),
            ("oscillating", "rectangle", "commit_direction", "reverse", "dynamic", "oscillation_requires_commitment", "48"),
            ("jump_loop", "circle", "extend_runway", "reverse", "subgoal", "failed_jump_requires_new_takeoff", "52"),
            ("transform_blocked", "rectangle", "flatten_reposition", "flatten", "none", "blocked_transform_requires_clearance", "52"),
            ("partner_deadlock", "circle", "yield_and_reassess", "reverse", "dynamic", "social_deadlock_requires_yield", "55"),
            ("partner_deadlock", "rectangle", "yield_and_reassess", "reverse", "dynamic", "social_deadlock_requires_yield", "55"),
            ("no_target_progress", "circle", "reverse_and_reassess", "jump", "opposite", "unsolved_state_requires_new_approach", "60"),
            ("no_target_progress", "rectangle", "reverse_and_reassess", "reverse", "opposite", "unsolved_state_requires_new_approach", "60"),
        )
        for values in recovery_schemas:
            model.decmem.add(
                actr.makechunk(
                    typename="recovery_schema",
                    stuck_kind=values[0],
                    shape=values[1],
                    strategy=values[2],
                    action=values[3],
                    direction=values[4],
                    association=values[5],
                    priority=values[6],
                )
            )

        model._explicit_declarative_chunks = set(model.decmem.keys())
    @staticmethod
    def _define_productions(model) -> None:
        # Production → adapter → production retrieval cycle.  Adapter states are
        # explicit controller states, not custom ACT-R modules.
        model.productionstring(name="P01_identify_self", utility=30, string="""
            =g> isa controller state identify_request
            ==>
            =g> isa controller state adapter_identify
        """)
        model.productionstring(name="P02_retrieve_identity", utility=29, string="""
            =g> isa controller state identity_lookup
            =imaginal_self> isa self_model identity =identity shape =shape
            ==>
            =g> isa controller state identity_wait
            +retrieval> isa identity_model identity =identity shape =shape
        """)
        model.productionstring(name="P03_confirm_identity", utility=28, string="""
            =g> isa controller state identity_wait
            =retrieval> isa identity_model identity =identity shape =shape status identified
            ?retrieval> state free buffer full
            ==>
            =g> isa controller state scan_request self_symbol =identity self_shape =shape
        """)
        model.productionstring(name="P04_request_situational_assessment", utility=28, string="""
            =g> isa controller state scan_request
            ==>
            =g> isa controller state adapter_assess
        """)
        model.productionstring(name="P05_retrieve_coordination_association", utility=27, string="""
            =g> isa controller state coordination_lookup
            =imaginal_coordination> isa coordination_model phase =phase
            ==>
            =g> isa controller state coordination_wait coordination_phase =phase
            +retrieval> isa coordination_schema phase =phase
        """)
        model.productionstring(name="P06_coordination_retrieved", utility=26, string="""
            =g> isa controller state coordination_wait
            =retrieval> isa coordination_schema phase =phase recommended_strategy =strategy
            ?retrieval> state free buffer full
            ==>
            =g> isa controller state adapter_coordination strategy =strategy coordination_phase =phase
        """)
        model.productionstring(name="P07_coordination_retrieval_failed", utility=4, string="""
            =g> isa controller state coordination_wait
            ?retrieval> state error
            ==>
            =g> isa controller state strategy_lookup
        """)
        model.productionstring(name="P10_retrieve_strategy_schema", utility=25, string="""
            =g> isa controller state strategy_lookup strategy =strategy
            ==>
            =g> isa controller state strategy_wait
            +retrieval> isa strategy_schema strategy =strategy
        """)
        model.productionstring(name="P11_strategy_schema_retrieved", utility=24, string="""
            =g> isa controller state strategy_wait strategy =strategy
            =retrieval> isa strategy_schema strategy =strategy
            ?retrieval> state free buffer full
            ==>
            =g> isa controller state adapter_strategy
        """)
        model.productionstring(name="P12_strategy_retrieval_failed", utility=3, string="""
            =g> isa controller state strategy_wait
            ?retrieval> state error
            ==>
            =g> isa controller state scan_request
        """)
        model.productionstring(name="P13_retrieve_recovery_schema", utility=34, string="""
            =g> isa controller state recovery_lookup self_shape =shape
            =imaginal_progress> isa progress_model stuck yes stuck_kind =kind
            ==>
            =g> isa controller state recovery_wait
            +retrieval> isa recovery_schema stuck_kind =kind shape =shape
        """)
        model.productionstring(name="P14_recovery_schema_retrieved", utility=33, string="""
            =g> isa controller state recovery_wait
            =retrieval> isa recovery_schema stuck_kind =kind strategy =strategy action =action
            ?retrieval> state free buffer full
            ==>
            =g> isa controller state adapter_recovery strategy =strategy
        """)
        model.productionstring(name="P15_recovery_retrieval_failed", utility=5, string="""
            =g> isa controller state recovery_wait
            ?retrieval> state error
            ==>
            =g> isa controller state scan_request
        """)
        model.productionstring(name="P16_adapter_error_recover", utility=40, string="""
            =g> isa controller state adapter_error
            ==>
            =g> isa controller state scan_request last_action adapter_error_recover
        """)

        actions = (
            ("P21_cooperative_raise_rectangle", "transform_up", "W", 21),
            ("P26_flatten_rectangle", "transform_down", "S", 17),
            ("P27_move_left", "move_left", "A", 14),
            ("P28_move_right", "move_right", "D", 14),
            ("P29_fast_fall", "fast_fall", "S", 16),
        )
        for name, flag, key, utility in actions:
            model.productionstring(name=name, utility=utility, string=f"""
                =g> isa controller state decide
                =imaginal_affordance> isa affordance_model {flag} yes
                ?manual> state free
                ==>
                =g> isa controller state await_motor last_action {flag}
                +manual> isa _manual cmd press_key key {key}
            """)

        jump_primes = (
            ("P22_prime_cooperative_jump_left", "cooperative_circle", "left", "A", 20),
            ("P23_prime_cooperative_jump_right", "cooperative_circle", "right", "D", 20),
            ("P24_prime_direct_jump_left", "direct_circle", "left", "A", 18),
            ("P25_prime_direct_jump_right", "direct_circle", "right", "D", 18),
        )
        for name, strategy, direction, key, utility in jump_primes:
            model.productionstring(name=name, utility=utility, string=f"""
                =g> isa controller state decide strategy {strategy}
                =imaginal_affordance> isa affordance_model prime_jump_{direction} yes
                ?manual> state free
                ==>
                =g> isa controller state jump_primed_{direction} last_action prime_jump_{direction}
                +manual> isa _manual cmd press_key key {key}
            """)

        social_actions = (
            ("P34_yield_route_left", "yield_route", "yield_direction", "left", "A", "yield_left", 32),
            ("P35_yield_route_right", "yield_route", "yield_direction", "right", "D", "yield_right", 32),
            ("P36_reunite_left", "reunite_team", "partner_direction", "left", "A", "reunite_left", 31),
            ("P37_reunite_right", "reunite_team", "partner_direction", "right", "D", "reunite_right", 31),
        )
        for name, strategy, slot, value, key, action, utility in social_actions:
            model.productionstring(name=name, utility=utility, string=f"""
                =g> isa controller state decide strategy {strategy}
                =imaginal_coordination> isa coordination_model {slot} {value}
                ?manual> state free
                ==>
                =g> isa controller state await_motor last_action {action}
                +manual> isa _manual cmd press_key key {key}
            """)

        model.productionstring(name="P38_reunite_wait_vertical_transition", utility=12, string="""
            =g> isa controller state decide strategy reunite_team
            =imaginal_coordination> isa coordination_model partner_direction level
            ==>
            =g> isa controller state wait_cycle last_action monitor_reunion
        """)
        model.productionstring(name="P30_wait_and_monitor", utility=6, string="""
            =g> isa controller state decide
            =imaginal_affordance> isa affordance_model wait yes
            ==>
            =g> isa controller state wait_cycle last_action wait
        """)
        model.productionstring(name="P31_replan", utility=2, string="""
            =g> isa controller state decide
            =imaginal_affordance> isa affordance_model replan yes
            ==>
            =g> isa controller state scan_request last_action replan
        """)
        model.productionstring(name="P32_jump_after_left_prime", utility=23, string="""
            =g> isa controller state jump_primed_left
            ?manual> state free
            ==>
            =g> isa controller state await_motor last_action jump
            +manual> isa _manual cmd press_key key W
        """)
        model.productionstring(name="P33_jump_after_right_prime", utility=23, string="""
            =g> isa controller state jump_primed_right
            ?manual> state free
            ==>
            =g> isa controller state await_motor last_action jump
            +manual> isa _manual cmd press_key key W
        """)

        recovery_actions = (
            ("P40_recovery_reverse_left", "left", "A", "recover_left", 38),
            ("P41_recovery_reverse_right", "right", "D", "recover_right", 38),
        )
        for name, direction, key, action, utility in recovery_actions:
            model.productionstring(name=name, utility=utility, string=f"""
                =g> isa controller state recover_decide
                =imaginal_recovery> isa recovery_plan action reverse direction {direction} status ready
                ?manual> state free
                ==>
                =g> isa controller state await_motor last_action {action}
                +manual> isa _manual cmd press_key key {key}
            """)
        model.productionstring(name="P42_recovery_prime_jump_left", utility=39, string="""
            =g> isa controller state recover_decide
            =imaginal_recovery> isa recovery_plan action jump direction left status ready
            ?manual> state free
            ==>
            =g> isa controller state recovery_jump_primed_left last_action recovery_prime_left
            +manual> isa _manual cmd press_key key A
        """)
        model.productionstring(name="P43_recovery_prime_jump_right", utility=39, string="""
            =g> isa controller state recover_decide
            =imaginal_recovery> isa recovery_plan action jump direction right status ready
            ?manual> state free
            ==>
            =g> isa controller state recovery_jump_primed_right last_action recovery_prime_right
            +manual> isa _manual cmd press_key key D
        """)
        model.productionstring(name="P44_recovery_jump_after_left_prime", utility=40, string="""
            =g> isa controller state recovery_jump_primed_left
            ?manual> state free
            ==>
            =g> isa controller state await_motor last_action recovery_jump
            +manual> isa _manual cmd press_key key W
        """)
        model.productionstring(name="P45_recovery_jump_after_right_prime", utility=40, string="""
            =g> isa controller state recovery_jump_primed_right
            ?manual> state free
            ==>
            =g> isa controller state await_motor last_action recovery_jump
            +manual> isa _manual cmd press_key key W
        """)
        model.productionstring(name="P46_recovery_flatten", utility=39, string="""
            =g> isa controller state recover_decide
            =imaginal_recovery> isa recovery_plan action flatten status ready
            ?manual> state free
            ==>
            =g> isa controller state await_motor last_action recovery_flatten
            +manual> isa _manual cmd press_key key S
        """)
        # Recovery never resets the level; it must continue through cognition.
        model.productionstring(name="P90_motor_complete_reassess", utility=24, string="""
            =g> isa controller state await_motor
            ?manual> state free
            ==>
            =g> isa controller state scan_request
        """)
        model.productionstring(name="P91_wait_cycle_reassess", utility=5, string="""
            =g> isa controller state wait_cycle
            ==>
            =g> isa controller state scan_request
        """)
