def walk_through_tree(node):
    if hasattr(node, "children"):
        yield node
        try:
            children = node.children
            yield from walk_through_tree(children[0])
            yield from walk_through_tree(children[1])
        except AttributeError:
            pass


def calculate_impurity(node, class_labels):
    class_weights = [node.stats[class_label] for class_label in node.stats.keys()]
    n_total = sum(class_weights)
    class_portions = [class_weight / n_total for class_weight in class_weights]
    class_impurities = [class_portion * (1 - class_portion) for class_portion in class_portions]
    impurity = sum(class_impurities)
    return impurity


class MeanDecreaseImpurityExplainer:

    def __init__(self, feature_names, tree_classifier):
        self.feature_names = feature_names
        self.model = tree_classifier

    def explain_one(self):
        importance_scores = {feature: 0. for feature in self.feature_names}
        split_nodes = {feature: 0 for feature in self.feature_names}
        root_node = self.model._root
        n_total_root = root_node.total_weight
        splits = iter(walk_through_tree(root_node))
        splits = list(splits)
        if len(splits) > 0:
            class_labels = self.model.classes
            for split_node in splits:
                feature = split_node.feature
                split_nodes[feature] += 1
                node_impurity = calculate_impurity(split_node, class_labels)
                n_total = split_node.total_weight
                children = split_node.children
                children_impurity = [calculate_impurity(child, class_labels) * child.total_weight / n_total for child in children]
                decrease_in_impurity = node_impurity - sum(children_impurity)
                weighted_decrease_in_impurity = decrease_in_impurity * n_total / n_total_root
                importance_scores[feature] += weighted_decrease_in_impurity
        return importance_scores, split_nodes
