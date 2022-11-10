package org.processmining.onlineconformance.stream;

import gnu.trove.map.TObjectShortMap;
import gnu.trove.map.hash.TObjectShortHashMap;

import org.processmining.contexts.uitopia.annotations.UITopiaVariant;
import org.processmining.eventstream.core.interfaces.XSEventStream;
import org.processmining.framework.plugin.PluginContext;
import org.processmining.framework.plugin.annotations.Plugin;
import org.processmining.framework.plugin.annotations.PluginVariant;
import org.processmining.plugins.etm.model.narytree.Configuration;
import org.processmining.plugins.etm.model.narytree.NAryTree;
import org.processmining.plugins.etm.model.narytree.TreeUtils;

@Plugin(name = "Online Replayer", parameterLabels = { "Stream", "ProcessTree" }, returnLabels = { "Online Replayer" }, returnTypes = { OnlineReplayerAlgorithm.class })
public class OnlineReplayerPlugin {

	@UITopiaVariant(affiliation = "TU/e", author = "B.F. van Dongen", email = "B.F.v.Dongen@tue.nl")
	@PluginVariant(requiredParameterLabels = { 0, 1 })
	public static OnlineReplayerAlgorithm onlineReplayerPlugin(PluginContext context, XSEventStream stream,
			NAryTree tree) {

		String[] activities = new String[] { "a", "b", "c", "d", "e", "f", "g", "h" };

		TObjectShortMap<String> map = new TObjectShortHashMap<String>();
		// initialize a tree with all node costs 5 for leafs and 0 otherwise.
		for (short i = 0; i < activities.length; i++) {
			map.put(activities[i], i);
		}

		OnlineReplayerAlgorithm result = new OnlineReplayerAlgorithm(tree, map);
		stream.connect(result);
		result.start();

		return result;
	}

	@UITopiaVariant(affiliation = "TU/e", author = "B.F. van Dongen", email = "B.F.v.Dongen@tue.nl")
	@PluginVariant(requiredParameterLabels = { 0 })
	public static OnlineReplayerAlgorithm onlineReplayerPlugin(PluginContext context, XSEventStream stream) {

		String[] activities = new String[] { "a", "b", "c", "d", "e", "f", "g", "h" };
		TObjectShortMap<String> map = new TObjectShortHashMap<String>();
		// initialize a tree with all node costs 5 for leafs and 0 otherwise.
		for (short i = 0; i < activities.length; i++) {
			map.put(activities[i], i);
		}

		NAryTree tree;
//		tree = TreeUtils.randomTree(activities.length, 0.2, 10, 50, new Random(3142));
		
		tree = TreeUtils
				.fromString(
						"SEQ( LEAF: a , XOR( LEAF: b , LEAF: c ) , LEAF: d ) [[ -, -, -, -, -, -] ]",
						map);

		tree = TreeUtils.flatten(tree);
		boolean[] b = new boolean[tree.size()];
		boolean[] h = new boolean[tree.size()];
		Configuration c = new Configuration(b, h);
		tree.addConfiguration(c);
		
		System.out.println(TreeUtils.toString(tree));

		return onlineReplayerPlugin(context, stream, tree);
	}

}
