#include "colors.inc"
#include "textures.inc"
#include "stones.inc"
#include "shapes.inc"
#include "glass.inc"
#include "metals.inc"
#include "woods.inc"

background { color Cyan }

camera {
  location <0,0,-10>
  look_at <0,0,0>
}

light_source { <5, 30, -30> color White }
light_source {<-5, 30, -30> color White }

// A flat circle mark
#declare mark = plane {
  <0, 0, -1>, 0
  clipped_by { cylinder { <0, 0, -1>, <0, 0, 1>, 1 }}
}

#declare dot_mark = object {
  mark
  pigment {
    image_map {
      png "dot_mark.png"
    }
    translate <0.5, 0.5, 0>
    scale 2
  }
}

#declare noise_mark = object {
  mark
  pigment {
    image_map {
      png "noise_mark.png"
    }
    translate <0.5, 0.5, 0>
    scale 2
  }
}

#declare plus_mark = object {
  mark
  pigment {
    image_map {
      png "plus_mark.png"
    }
    translate <0.5, 0.5, 0>
    scale 2
  }
}

#declare quarter_mark = object {
  mark
  pigment {
    image_map {
      png "quarter_mark.png"
    }
    translate <0.5, 0.5, 0>
    scale 2
  }
}

// Checkered plane for reference
plane {
  <0,1,0>, -1.5
  pigment { 
    checker color Gray35, color Gray65
  }
}

object {
  dot_mark
  translate <-4, 0, 0>
}

object { 
  noise_mark
  translate <-2, 0, 0>
}

object { 
  plus_mark
  translate <0, 0, 0>
}

object { 
  quarter_mark
  translate <2, 0, 0>
}
