"""축소 3-way merge와 textconv 요약 (기획서 §10.2)."""

from conftest import MINIVIT
from torchflow import merge as merger
from torchflow.ir import load


def three():
    return load(MINIVIT), load(MINIVIT), load(MINIVIT)


def test_changes_in_different_places_merge_by_themselves():
    base, ours, theirs = three()
    ours.graph.instances["01J9I103"].args["in_features"] = 256
    theirs.hparams["depth"].default = 12

    merged, conflicts = merger.merge(base, ours, theirs)
    assert conflicts == []
    assert merged.graph.instances["01J9I103"].args["in_features"] == 256
    assert merged.hparams["depth"].default == 12


def test_the_same_field_changed_two_ways_is_a_conflict():
    base, ours, theirs = three()
    ours.graph.instances["01J9I103"].args["in_features"] = 256
    theirs.graph.instances["01J9I103"].args["in_features"] = 512

    merged, conflicts = merger.merge(base, ours, theirs)
    assert len(conflicts) == 1 and "in_features" in conflicts[0]
    assert merged.graph.instances["01J9I103"].args["in_features"] == 256, "우리 값을 두고 사람에게 넘긴다"


def test_a_block_added_on_the_other_branch_comes_along():
    from torchflow.ir import Instance, Node

    base, ours, theirs = three()
    theirs.graph.instances["new"] = Instance(label="drop", type="torch.nn.Dropout", args={"p": 0.1})
    theirs.graph.nodes.append(Node(id="N-new", label="drop", call="new", method="forward"))
    theirs.graph.edges.append(("N-new.output", "01J9Q4B6.input"))
    ours.graph.instances["01J9I103"].args["in_features"] = 256

    merged, conflicts = merger.merge(base, ours, theirs)
    assert conflicts == []
    assert any(node.id == "N-new" for node in merged.graph.nodes)
    assert ("N-new.output", "01J9Q4B6.input") in [tuple(edge) for edge in merged.graph.edges]
    assert merged.graph.instances["01J9I103"].args["in_features"] == 256


def test_a_block_removed_on_one_side_stays_removed():
    base, ours, theirs = three()
    theirs.graph.nodes = [node for node in theirs.graph.nodes if node.id != "01J9Q4B4"]
    theirs.graph.edges = [edge for edge in theirs.graph.edges
                          if "01J9Q4B4" not in edge[0] and "01J9Q4B4" not in edge[1]]

    merged, conflicts = merger.merge(base, ours, theirs)
    assert conflicts == []
    assert not any(node.id == "01J9Q4B4" for node in merged.graph.nodes)
    assert not any("01J9Q4B4" in edge[0] or "01J9Q4B4" in edge[1] for edge in merged.graph.edges)


def test_removing_what_the_other_side_edited_is_a_conflict():
    base, ours, theirs = three()
    ours.graph.nodes = [node for node in ours.graph.nodes if node.id != "01J9Q4B2"]
    theirs.graph.nodes[1].args["extra"] = 1
    theirs.graph.nodes = [node for node in theirs.graph.nodes]
    for node in theirs.graph.nodes:
        if node.id == "01J9Q4B2":
            node.label = "renamed"

    _, conflicts = merger.merge(base, theirs, ours)
    assert any("지우고" in conflict for conflict in conflicts)


def test_summary_is_readable_and_stable():
    text = merger.summary(load(MINIVIT))
    assert text.startswith("graph MiniViT")
    assert "[graph]" in text and "->" in text
    assert text == merger.summary(load(MINIVIT))
