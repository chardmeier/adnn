#ifndef NNET_VOCMAP_H
#define NNET_VOCMAP_H

#include <unordered_map>

namespace vocmap {

struct vocmap {
	typedef std::unordered_map<std::string,voc_id> map_type;

	enum { UNKNOWN_WORD = 0 };

	voc_id maxid;
	map_type map;

	vocmap() : maxid(1) {
		map.insert(std::make_pair("<unk>", UNKNOWN_WORD));
	}
};

voc_id voc_lookup(const std::string &word, vocmap &voc, bool extend = false) {
	voc_id id;
	vocmap::map_type::const_iterator it = voc.map.find(word);
	if(it != voc.map.end()) 
		id = it->second;
	else {
		if(extend) {
			id = voc.maxid++;
			voc.map.insert(std::make_pair(word, id));
		} else
			id = vocmap::UNKNOWN_WORD;
	}

	return id;
}

} // namespace vocmap

#endif
