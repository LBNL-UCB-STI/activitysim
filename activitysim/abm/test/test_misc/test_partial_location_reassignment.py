from types import SimpleNamespace

import pandas as pd

from activitysim.abm.models import location_choice


class _FakeTracing:
    def trace_df(self, *args, **kwargs):
        return None


class _FakeState:
    def __init__(self):
        self.settings = SimpleNamespace(
            want_dest_choice_sample_tables=False,
            trace_hh_id=False,
        )
        self.tables = {}
        self.tracing = _FakeTracing()

    def add_table(self, name, table):
        self.tables[name] = table.copy() if hasattr(table, "copy") else table

    def is_table(self, name):
        return name in self.tables

    def extend_table(self, name, table):
        self.tables[name] = table.copy() if hasattr(table, "copy") else table


class _FakeShadowPriceCalculator:
    def __init__(self):
        self.use_shadow_pricing = False
        self.max_iterations = 1
        self.modeled_size = None
        self.movable_chooser_ids = None
        self.choices = None
        self.segment_ids = None

    def set_movable_choosers(self, chooser_index):
        self.movable_chooser_ids = pd.Index(chooser_index).copy()

    def set_choices(self, choices, segment_ids):
        self.choices = choices.copy()
        self.segment_ids = segment_ids.copy()


def _model_settings(reassign_filter_column_name=None):
    return SimpleNamespace(
        CHOOSER_FILTER_COLUMN_NAME="is_student",
        REASSIGN_FILTER_COLUMN_NAME=reassign_filter_column_name,
        DEST_CHOICE_COLUMN_NAME="school_zone_id",
        DEST_CHOICE_LOGSUM_COLUMN_NAME="school_location_logsum",
        MODE_CHOICE_LOGSUM_COLUMN_NAME="school_mode_choice_logsum",
        DEST_CHOICE_SAMPLE_TABLE_NAME=None,
        CHOOSER_SEGMENT_COLUMN_NAME="school_segment",
        SEGMENT_IDS={"gradeschool": 1},
        SHADOW_PRICE_TABLE=None,
        MODELED_SIZE_TABLE=None,
    )


def _persons_tables():
    persons = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "is_student": [True, True, True, False],
            "needs_reassignment": [True, False, True, True],
            "school_segment": [1, 1, 1, 1],
            "school_zone_id": [10, 20, 30, 40],
            "school_location_logsum": [1.0, 2.0, 3.0, 4.0],
            "school_mode_choice_logsum": [11.0, 22.0, 33.0, 44.0],
        }
    ).set_index("person_id")

    persons_merged = persons.copy()

    return persons, persons_merged


def test_iterate_location_choice_preserves_existing_destinations_and_logsums(
    monkeypatch,
):
    state = _FakeState()
    persons, persons_merged = _persons_tables()
    spc = _FakeShadowPriceCalculator()

    def fake_run_location_choice(
        _state,
        choosers,
        _network_los,
        shadow_price_calculator,
        want_logsums,
        want_sample_table,
        estimator,
        model_settings,
        chunk_size,
        chunk_tag,
        trace_label,
    ):
        assert shadow_price_calculator is spc
        assert want_logsums
        assert not want_sample_table
        assert list(choosers.index) == [1, 3]
        choices = pd.DataFrame(
            {
                "choice": [101, 103],
                "logsum": [1.5, 3.5],
                location_choice.ALT_LOGSUM: [11.5, 33.5],
            },
            index=choosers.index,
        )
        return choices, None

    monkeypatch.setattr(
        location_choice.shadow_pricing,
        "load_shadow_price_calculator",
        lambda *_args, **_kwargs: spc,
    )
    monkeypatch.setattr(location_choice, "run_location_choice", fake_run_location_choice)
    monkeypatch.setattr(location_choice.expressions, "annotate_tables", lambda *a, **k: None)

    location_choice.iterate_location_choice(
        state=state,
        model_settings=_model_settings("needs_reassignment"),
        persons_merged=persons_merged,
        persons=persons.copy(),
        households=pd.DataFrame(),
        network_los=None,
        estimator=None,
        chunk_size=0,
        locutor=False,
        trace_label="school_location",
    )

    updated_persons = state.tables["persons"]

    assert updated_persons["school_zone_id"].to_dict() == {1: 101, 2: 20, 3: 103, 4: -1}
    assert updated_persons.loc[[1, 2, 3], "school_location_logsum"].to_dict() == {
        1: 1.5,
        2: 2.0,
        3: 3.5,
    }
    assert pd.isna(updated_persons.loc[4, "school_location_logsum"])
    assert updated_persons.loc[[1, 2, 3], "school_mode_choice_logsum"].to_dict() == {
        1: 11.5,
        2: 22.0,
        3: 33.5,
    }
    assert pd.isna(updated_persons.loc[4, "school_mode_choice_logsum"])
    assert spc.movable_chooser_ids.tolist() == [1, 3]
    assert spc.choices.to_dict() == {1: 101, 2: 20, 3: 103}
    assert spc.segment_ids.to_dict() == {1: 1, 2: 1, 3: 1}


def test_iterate_location_choice_without_reassign_filter_preserves_current_behavior(
    monkeypatch,
):
    state = _FakeState()
    persons, persons_merged = _persons_tables()
    spc = _FakeShadowPriceCalculator()

    def fake_run_location_choice(
        _state,
        choosers,
        _network_los,
        shadow_price_calculator,
        want_logsums,
        want_sample_table,
        estimator,
        model_settings,
        chunk_size,
        chunk_tag,
        trace_label,
    ):
        assert shadow_price_calculator is spc
        assert list(choosers.index) == [1, 2, 3]
        choices = pd.DataFrame(
            {
                "choice": [101, 102, 103],
                "logsum": [1.5, 2.5, 3.5],
                location_choice.ALT_LOGSUM: [11.5, 22.5, 33.5],
            },
            index=choosers.index,
        )
        return choices, None

    monkeypatch.setattr(
        location_choice.shadow_pricing,
        "load_shadow_price_calculator",
        lambda *_args, **_kwargs: spc,
    )
    monkeypatch.setattr(location_choice, "run_location_choice", fake_run_location_choice)
    monkeypatch.setattr(location_choice.expressions, "annotate_tables", lambda *a, **k: None)

    location_choice.iterate_location_choice(
        state=state,
        model_settings=_model_settings(),
        persons_merged=persons_merged,
        persons=persons.copy(),
        households=pd.DataFrame(),
        network_los=None,
        estimator=None,
        chunk_size=0,
        locutor=False,
        trace_label="school_location",
    )

    updated_persons = state.tables["persons"]

    assert updated_persons["school_zone_id"].to_dict() == {1: 101, 2: 102, 3: 103, 4: -1}
    assert updated_persons.loc[[1, 2, 3], "school_location_logsum"].to_dict() == {
        1: 1.5,
        2: 2.5,
        3: 3.5,
    }
    assert pd.isna(updated_persons.loc[4, "school_location_logsum"])
    assert updated_persons.loc[[1, 2, 3], "school_mode_choice_logsum"].to_dict() == {
        1: 11.5,
        2: 22.5,
        3: 33.5,
    }
    assert pd.isna(updated_persons.loc[4, "school_mode_choice_logsum"])
    assert spc.movable_chooser_ids.tolist() == [1, 2, 3]
    assert spc.choices.to_dict() == {1: 101, 2: 102, 3: 103}
