file(REMOVE_RECURSE
  "release/libtokenizers_c.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/tokenizers_c.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
