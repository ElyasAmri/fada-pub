package com.fada.ultrasound.data

object UltrasoundClasses {

    /**
     * Class labels in model output order (indices 0-11).
     * These match the training data folder names.
     */
    val CLASSES: List<String> = listOf(
        "Abodomen",                      // 0
        "Aorta",                         // 1
        "Cervical",                      // 2
        "Cervix",                        // 3
        "Femur",                         // 4
        "Non_standard_NT",               // 5
        "Public_Symphysis_fetal_head",   // 6
        "Standard_NT",                   // 7
        "Thorax",                        // 8
        "Trans-cerebellum",              // 9
        "Trans-thalamic",                // 10
        "Trans-ventricular"              // 11
    )

    /**
     * User-friendly display names (corrects typos).
     */
    val DISPLAY_NAMES: Map<String, String> = mapOf(
        "Abodomen" to "Abdomen",
        "Aorta" to "Aortic Arch",
        "Cervical" to "Cervical View",
        "Cervix" to "Cervix",
        "Femur" to "Femur",
        "Non_standard_NT" to "Non-standard NT",
        "Public_Symphysis_fetal_head" to "Fetal Head Position",
        "Standard_NT" to "Standard NT",
        "Thorax" to "Thorax",
        "Trans-cerebellum" to "Transcerebellar Plane",
        "Trans-thalamic" to "Transthalamic Plane",
        "Trans-ventricular" to "Transventricular Plane"
    )

    /**
     * Clinical descriptions for each class.
     */
    val DESCRIPTIONS: Map<String, String> = mapOf(
        "Abodomen" to "Abdominal cross-section showing stomach, liver, and cord insertion",
        "Aorta" to "Aortic arch view for cardiac output assessment",
        "Cervical" to "Cervical view for cervix evaluation",
        "Cervix" to "Direct cervix view for length measurement",
        "Femur" to "Femur length measurement for growth assessment",
        "Non_standard_NT" to "Non-standard nuchal translucency view",
        "Public_Symphysis_fetal_head" to "Fetal head position relative to pubic symphysis",
        "Standard_NT" to "Standard nuchal translucency measurement",
        "Thorax" to "Thoracic cross-section showing lung fields and diaphragm",
        "Trans-cerebellum" to "Transcerebellar plane for posterior fossa evaluation",
        "Trans-thalamic" to "Transthalamic plane for midline brain structures",
        "Trans-ventricular" to "Transventricular plane for ventricle measurement"
    )

    /**
     * Get display name for a class label.
     */
    fun getDisplayName(className: String): String {
        return DISPLAY_NAMES[className] ?: className
    }

    /**
     * Get display name by class index.
     */
    fun getDisplayNameByIndex(index: Int): String {
        if (index < 0 || index >= CLASSES.size) return "Unknown"
        return getDisplayName(CLASSES[index])
    }

    /**
     * Get clinical description for a class.
     */
    fun getDescription(className: String): String {
        return DESCRIPTIONS[className] ?: "No description available"
    }

    /**
     * Get description by class index.
     */
    fun getDescriptionByIndex(index: Int): String {
        if (index < 0 || index >= CLASSES.size) return "No description available"
        return getDescription(CLASSES[index])
    }

    /**
     * Number of classes in the model.
     */
    val NUM_CLASSES: Int = CLASSES.size
}
