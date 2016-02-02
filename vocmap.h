#ifndef NNET_VOCMAP_H
#define NNET_VOCMAP_H

#include <unordered_map>

namespace vocmap {

typedef unsigned long voc_id;

class vocmap {
private:
	typedef std::unordered_map<std::string,voc_id> map_type;

	voc_id maxid;
	map_type map;

public:
	enum { UNKNOWN_WORD = 0 };

	vocmap() : maxid(1) {
		map.insert(std::make_pair("<unk>", UNKNOWN_WORD));
	}

	voc_id lookup(const std::string &word, bool extend = false);

	std::size_t size() const {
		return maxid;
	}
};

voc_id vocmap::lookup(const std::string &word, bool extend) {
	voc_id id;
	vocmap::map_type::const_iterator it = map.find(word);
	if(it != map.end()) 
		id = it->second;
	else {
		if(extend) {
			id = maxid++;
			map.insert(std::make_pair(word, id));
		} else
			id = UNKNOWN_WORD;
	}

	return id;
}

} // namespace vocmap

#endif
