package ee.ut.cs.dsg.confcheck.util;

import ee.ut.cs.dsg.confcheck.State;

import java.lang.reflect.Constructor;
import java.lang.reflect.Method;

public class JavaClassLoader extends ClassLoader {
    public Class<?> loadedMyClass;
    public Object myClassObject;

    public void invokeClass(String classBinName, Class<?>[] type, Object[] params) {

        try {
            ClassLoader classLoader = this.getClass().getClassLoader();
            loadedMyClass = classLoader.loadClass(classBinName);
            Constructor<?> constructor = loadedMyClass.getDeclaredConstructor(type);
            myClassObject = constructor.newInstance(params);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void invokeCheck(Object[] params, Class<?>[] type) {

        try {
            Method method1 = loadedMyClass.getDeclaredMethod("check", type);
            method1.invoke(myClassObject, params);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public State invokeGetCurrentOptimalState(Object[] params, Class<?>[] type) {
        State state = null;

        try {
            Method method1 = loadedMyClass.getDeclaredMethod("getCurrentOptimalState", type);
            state = (State) method1.invoke(myClassObject, params);

        } catch (Exception e) {
            e.printStackTrace();
        }
        return state;
    }
}
