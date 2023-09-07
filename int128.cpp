#include <ostream>

#include "int128.h"

const uint128_pod kuint128max = { static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL),
                                 static_cast<uint64_t>(0xFFFFFFFFFFFFFFFFULL) };

std::ostream& operator<<(std::ostream& o, const uint128& b) {
    return (o << b.hi_ << "::" << b.lo_);
}