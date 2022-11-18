package algorithm;

import org.deckfour.xes.info.XLogInfo;
import org.deckfour.xes.info.XLogInfoFactory;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;
import org.processmining.contexts.uitopia.UIPluginContext;
import org.processmining.framework.plugin.annotations.Plugin;
import org.processmining.framework.util.Pair;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.semantics.petrinet.Marking;
import org.processmining.streamconformance.local.model.LocalConformanceTracker;
import org.processmining.streamconformance.local.model.LocalModelStructure;
import org.processmining.streamconformance.local.model.OnlineConformanceScore;
import org.processmining.streamconformance.utils.PetrinetHelper;
import org.processmining.streamconformance.utils.XLogHelper;

// added
import org.processmining.plugins.petrinet.behavioralanalysis.CGGenerator;
import org.processmining.models.graphbased.directed.transitionsystem.CoverabilityGraph;

public class LocalOfflineConformance {
    public static XLog plugin(Plugin context , Petrinet net, Marking initMarking, XLog log) throws Exception {

        // build coverability graphs
        CoverabilityGraph coverabilityGraph = context.tryToFindOrConstructFirstNamedObject(
                CoverabilityGraph.class,
                CGGenerator.class.getAnnotation(Plugin.class).name(),
                null,
                null,
                net,
                initMarking);


        CoverabilityGraph coverabilityGraph1 = new CoverabilityGraph();

        // build coverability graph of unfolded net
        Pair<Petrinet, Marking> unfoldedTotal = PetrinetHelper.unfold(context, net);
        CoverabilityGraph coverabilityGraphUnfolded = context.tryToFindOrConstructFirstNamedObject(
                CoverabilityGraph.class,
                CGGenerator.class.getAnnotation(Plugin.class).name(),
                null,
                null,
                unfoldedTotal.getFirst(),
                unfoldedTotal.getSecond());

        // build coverability graph of dual net
        Pair<Petrinet, Marking> dualNet = PetrinetHelper.computeDual(context, net);
        Pair<Petrinet, Marking> unfoldedDualNet = PetrinetHelper.unfold(context, dualNet.getFirst());
        CoverabilityGraph coverabilityGraphDualUnfolded = context.tryToFindOrConstructFirstNamedObject(
                CoverabilityGraph.class,
                CGGenerator.class.getAnnotation(Plugin.class).name(),
                null,
                null,
                unfoldedDualNet.getFirst(),
                unfoldedDualNet.getSecond());

        //populateStructure(coverabilityGraph, coverabilityGraphUnfolded, coverabilityGraphDualUnfolded);

        LocalModelStructure lms =  LocalModelStructure(CoverabilityGraph coverabilityGraph, CoverabilityGraph coverabilityGraphUnfolded, CoverabilityGraph coverabilityGraphDualUnfolded);
        LocalConformanceTracker lct = new LocalConformanceTracker(lms, log.size());
        XLogInfo info = XLogInfoFactory.createLogInfo(log);

        context.getProgress().setMinimum(0);
        context.getProgress().setMaximum(info.getNumberOfEvents());

        XLog enriched = XLogHelper.generateNewXLog(XLogHelper.getName(log));
        for (XTrace origTrace : log) {
            String traceName = XLogHelper.getName(origTrace);
            XTrace newTrace = XLogHelper.createTrace(traceName);
            for (XEvent origEvent : origTrace) {
                String eventName = XLogHelper.getName(origEvent);
                OnlineConformanceScore ocs = lct.replayEvent(traceName, eventName);
                XEvent newEvent = XLogHelper.insertEvent(newTrace, eventName);
                XLogHelper.decorateElement(newEvent, "conformance", ocs.getConformance());
                XLogHelper.decorateElement(newEvent, "confidence", ocs.getConfidence());
                XLogHelper.decorateElement(newEvent, "completeness", ocs.getCompleteness());
                context.getProgress().inc();
            }
            enriched.add(newTrace);
        }

        return enriched;
    }
}
