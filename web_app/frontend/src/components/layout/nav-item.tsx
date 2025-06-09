import React from 'react';
import { Link } from 'react-router-dom';

const NavItem: React.FC<{name: string, href: string}> = ({name, href}) => {
  return (
    <li className="mx-4">
     <Link
  to={href}
  className="
    no-underline
    text-white
    py-2
    px-4
    block
    text-xl
    font-bold

    transition-transform
    duration-200
    ease-in-out
    hover:scale-110
    active:scale-110
  "
>
  {name}
</Link>
    </li>
  );
};

export default NavItem;