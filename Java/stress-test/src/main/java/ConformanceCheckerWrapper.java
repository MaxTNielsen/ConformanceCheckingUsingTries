import beamline.events.BEvent;
import beamline.models.algorithms.StreamMiningAlgorithm;
import ee.ut.cs.dsg.confcheck.C_3PO;
import ee.ut.cs.dsg.confcheck.State;
import ee.ut.cs.dsg.confcheck.alignment.Alignment;
import ee.ut.cs.dsg.confcheck.trie.Trie;
import ee.ut.cs.dsg.confcheck.util.AlphabetService;
import org.deckfour.xes.classification.XEventAttributeClassifier;
import org.deckfour.xes.classification.XEventClass;
import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.info.XLogInfo;
import org.deckfour.xes.info.XLogInfoFactory;
import org.deckfour.xes.info.impl.XLogInfoImpl;
import org.deckfour.xes.model.XEvent;
import org.deckfour.xes.model.XLog;
import org.deckfour.xes.model.XTrace;
import org.openjdk.jol.info.GraphLayout;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static ee.ut.cs.dsg.confcheck.Runner.loadLog;

public class ConformanceCheckerWrapper extends StreamMiningAlgorithm<OnlineConformanceResults> {
    private static final long serialVersionUID = 6287730078016220573L;
    private C_3PO checker;
    private AlphabetService service;
    private OnlineConformanceResults last;
    private Trie trie;

    private boolean isMemoryExperiment = false;

    public ConformanceCheckerWrapper(int logCost, int modelCost, int maxStatesInQueue, int maxTrials, boolean isStandardAlign, String costType, HashMap<String, String> urls, String log, boolean isWarmStartAllStates, String proxyPath) {
        init();
        this.last = new OnlineConformanceResults(service);
        this.trie = constructTrie(proxyPath);
        this.checker = new C_3PO(this.trie, logCost, modelCost, maxStatesInQueue, maxTrials, isStandardAlign, costType, urls, log, isWarmStartAllStates);
    }

    @Override
    public OnlineConformanceResults ingest(BEvent event) {
        State state;
        Alignment alg;
        long start;
        long executionTime;
        long memorySizeCases = 0;
        long memorySizeStates = 0;
        int totalCases = checker.sizeOfCasesInBuffer();
        int totalStates = checker.statesInBuffer(event.getTraceName());
        List<String> e = new ArrayList<>();
        e.add(Character.toString(service.alphabetize(event.getEventName())));
        start = System.nanoTime();
        this.checker.check(e, event.getTraceName());
        state = checker.getCurrentOptimalState(event.getTraceName(), false);
        alg = null;
        try {
            alg = state.getAlignment();
        } catch (NullPointerException except) {
            System.out.println("Optimal alignment state was not found");
        }
        executionTime = System.nanoTime() - start;
        if(isMemoryExperiment){
            memorySizeCases = GraphLayout.parseInstance(checker.getCasesInBuffer()).totalSize();
            memorySizeStates = GraphLayout.parseInstance(checker.getTracesInBuffer(event.getTraceName())).totalSize();
        }
        this.last.setLastEvent(event);
        this.last.setConformance(alg.getTotalCost());
        this.last.setConfidence(state.getNode().getConfidenceCost());
        this.last.setCompleteness(state.getCompletenessCost());
        this.last.setProcessingTime(executionTime);
        this.last.setTotalCases(totalCases);
        this.last.setTotalStates(totalStates);
        this.last.setAlgSize(alg.getMoves().size());
        this.last.setMemorySizeTraces(memorySizeStates);
        this.last.setMemorySizeCases(memorySizeCases);
        return this.last;
    }

    private void init() {
        service = new AlphabetService();
    }

    public Trie constructTrie(String inputProxyLogFile) {
        XLog inputProxyLog = loadLog(inputProxyLogFile);
        XEventClass dummyEvClass = new XEventClass("DUMMY", 99999);
        XEventClassifier eventClassifier = XLogInfoImpl.NAME_CLASSIFIER;

        try {
            //
            XEventClassifier attClassifier = null;
            if (inputProxyLog.getClassifiers().size() > 0) attClassifier = inputProxyLog.getClassifiers().get(0);
            else attClassifier = new XEventAttributeClassifier("concept:name", new String[]{"concept:name"});
            XLogInfo logInfo = XLogInfoFactory.createLogInfo(inputProxyLog, attClassifier);
            int count = 999;
            if (logInfo.getNameClasses().getClasses().size() > 0) {
                count = 0;
                for (XEventClass clazz : logInfo.getNameClasses().getClasses()) {
                    count++;
                }
            }

            Trie t = new Trie(count, service);
            List<String> templist;

            for (XTrace trace : inputProxyLog) {
                templist = new ArrayList<String>();
                for (XEvent e : trace) {
                    String label = e.getAttributes().get(attClassifier.getDefiningAttributeKeys()[0]).toString();

                    templist.add(Character.toString(service.alphabetize(label)));
                }
                if (templist.size() > 0) {

                    t.addTrace(templist);
                }
            }
            return t;
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }
}
