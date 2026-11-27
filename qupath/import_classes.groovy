/*
 * vollmilch -> QuPath: set the classification of existing cell objects from a vollmilch
 * classes.csv (written by `vollmilch export`). Use this when the cells are already in the
 * QuPath project (imported earlier from a vollmilch GeoJSON) and you only changed the gates.
 * Objects are matched by name (= vollmilch cell_id, e.g. "Biopsy_A:1234").
 *
 * For a first import, drag the <image>_cells.geojson onto the open image instead
 * (or File > Import objects); it carries outlines, classes and measurements.
 *
 * Tested with QuPath 0.5-0.7. Population names must not contain commas.
 */
def csvPath = "/path/to/vollmilch_out/export/gates/classes.csv"   // <- edit

def f = new File(csvPath)
if (!f.exists()) { println "File not found: $csvPath"; return }
def lines = f.readLines()
def header = lines[0].split(",", -1) as List
int iId = header.indexOf("cell_id"), iPop = header.indexOf("population")
def classes = [:]
lines.drop(1).each { l ->
    def p = l.split(",", -1)
    if (p.size() > Math.max(iId, iPop)) classes[p[iId]] = p[iPop]
}
int n = 0, cleared = 0
getDetectionObjects().each { o ->
    def name = o.getName()
    if (name != null && classes.containsKey(name)) {
        def c = classes[name]
        if (c) { o.setPathClass(getPathClass(c)); n++ } else { o.setPathClass(null); cleared++ }
    }
}
fireHierarchyUpdate()
println "vollmilch: classified $n objects, $cleared without population (of ${classes.size()} in the CSV)"
